"""Observability -- metrics, structured logs, and online calibration.

A rating engine can degrade without erroring. Predictions drift out of
calibration when the population shifts, the catalogue turns over, or an upstream
adapter starts sending a different outcome scale, and none of that raises an
exception. The offline ECE measured on synthetic data says nothing about any of
it. So the same metric that was used to justify the model has to be computed
continuously in production, on live traffic.

Three pieces:

``Metrics``          counters plus latency quantiles, in-process, cheap
``CalibrationMonitor`` sliding-window ECE over (predicted p, realised outcome)
``log_event``        one-line JSON to stdout, for whatever collects logs

Deliberately no dependency on a metrics vendor. The surface is small enough that
exporting to Prometheus/OTel is an adapter, and hard-wiring one now would make
the engine harder to embed than it needs to be.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Mapping

__all__ = ["Metrics", "CalibrationMonitor", "log_event", "Timer", "render_openmetrics"]



def log_event(event: str, **fields: Any) -> None:
    """One JSON object per line. Structured from the start because grepping
    free-form log prose is how incidents get longer."""
    rec = {"ts": round(time.time(), 3), "event": event}
    rec.update(fields)
    sys.stdout.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


@dataclass
class CalibrationMonitor:
    """Sliding-window Expected Calibration Error.

    ECE is the metric the model was selected on (0.0171 offline vs Elo's 0.0756),
    so it is the metric whose regression matters. The window is bounded so memory
    is constant and the number tracks *recent* behaviour -- a lifetime average
    hides a drift that started yesterday.

    ``alert_threshold`` is intentionally None by default. A shipped threshold
    would be an invented constant; it should be set from a tenant's own observed
    baseline once one exists.
    """

    window: int = 5000
    n_bins: int = 10
    alert_threshold: float | None = None
    _buf: Deque[tuple[float, float]] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record(self, p_hat: float, outcome: float) -> None:
        with self._lock:
            self._buf.append((float(p_hat), float(outcome)))
            while len(self._buf) > self.window:
                self._buf.popleft()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = list(self._buf)
        n = len(data)
        if n == 0:
            return {"n": 0, "ece": None, "brier": None, "bins": [], "alert": False}

        edges = [i / self.n_bins for i in range(self.n_bins + 1)]
        bins = []
        ece = 0.0
        brier = sum((p - o) ** 2 for p, o in data) / n
        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            sel = [(p, o) for p, o in data
                   if (p >= lo and p < hi) or (i == self.n_bins - 1 and p == hi)]
            if not sel:
                continue
            m = len(sel)
            mean_p = sum(p for p, _ in sel) / m
            mean_o = sum(o for _, o in sel) / m
            ece += (m / n) * abs(mean_p - mean_o)
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": m,
                         "p_hat": round(mean_p, 4), "actual": round(mean_o, 4),
                         "diff": round(mean_o - mean_p, 4)})

        alert = self.alert_threshold is not None and ece > self.alert_threshold
        return {"n": n, "ece": round(ece, 4), "brier": round(brier, 4),
                "bins": bins, "alert": alert}


class Metrics:
    """Counters, latency quantiles, calibration. In-process and per-process.

    ``max_series`` bounds label cardinality. Counters are labelled by tenant, so
    an unbounded map grows one permanent entry per tenant that ever sent a
    request; overflow is reported as a counter of its own rather than silently
    dropping data.
    """

    def __init__(self, latency_window: int = 2000, max_series: int = 4096) -> None:
        self._counters: dict[str, int] = {}
        self._lat: dict[str, Deque[float]] = {}
        self._window = latency_window
        self._max_series = max_series
        self._dropped = 0
        self._lock = threading.Lock()
        self.calibration = CalibrationMonitor()

    def incr(self, name: str, by: int = 1, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            if key not in self._counters and len(self._counters) >= self._max_series:
                self._dropped += 1
                return
            self._counters[key] = self._counters.get(key, 0) + by

    def observe_latency(self, name: str, ms: float) -> None:
        with self._lock:
            q = self._lat.get(name)
            if q is None:
                if len(self._lat) >= self._max_series:
                    return
                q = self._lat.setdefault(name, deque())
            q.append(ms)
            while len(q) > self._window:
                q.popleft()

    def snapshot(self, tenant: str | None = None) -> dict[str, Any]:
        """Full snapshot, or one tenant's view of it.

        Counters carry a ``tenant`` label, so an unfiltered snapshot is one
        tenant's window onto every other tenant's traffic volume, degradation
        rate and throttling. Passing ``tenant`` keeps that tenant's series plus
        the unlabelled process-level ones, which describe the process rather than
        anyone's data.
        """
        with self._lock:
            counters = dict(self._counters)
            lat = {k: sorted(v) for k, v in self._lat.items()}
            dropped = self._dropped
        if tenant is not None:
            counters = {k: v for k, v in counters.items() if _visible_to(k, tenant)}
            lat = {k: v for k, v in lat.items() if _visible_to(k, tenant)}
        latency = {}
        for name, vals in lat.items():
            if not vals:
                continue
            latency[name] = {
                "n": len(vals),
                "p50": round(_q(vals, 0.50), 2),
                "p95": round(_q(vals, 0.95), 2),
                "p99": round(_q(vals, 0.99), 2),
                "max": round(vals[-1], 2),
            }
        out = {"counters": counters, "latency_ms": latency,
               "calibration": self.calibration.snapshot()}
        if dropped:
            out["series_dropped"] = dropped
        return out


def _visible_to(key: str, tenant: str) -> bool:
    """Whether a series belongs to ``tenant`` or to no tenant at all."""
    _, labels = _split_key(key)
    owner = labels.get("tenant")
    return owner is None or owner == tenant



def _key(name: str, labels: Mapping[str, Any]) -> str:
    if not labels:
        return name
    tail = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
    return f"{name}{{{tail}}}"


def _split_key(key: str) -> tuple[str, dict[str, str]]:
    """Inverse of ``_key``: ``a{b=c}`` -> ``("a", {"b": "c"})``."""
    if not key.endswith("}") or "{" not in key:
        return key, {}
    name, _, rest = key.partition("{")
    labels: dict[str, str] = {}
    for part in rest[:-1].split(","):
        k, _, v = part.partition("=")
        if k:
            labels[k.strip()] = v.strip()
    return name, labels


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels_out(labels: Mapping[str, str], extra: Mapping[str, str] = {}) -> str:
    merged = {**labels, **extra}
    if not merged:
        return ""
    body = ",".join(f'{k}="{_escape(str(merged[k]))}"' for k in merged)
    return "{" + body + "}"


def render_openmetrics(snapshot: Mapping[str, Any], prefix: str = "engine") -> str:
    """Render ``Metrics.snapshot()`` as OpenMetrics/Prometheus text.

    Kept as a pure function of the snapshot dict so the metrics core stays
    vendor-free: swapping exposition formats does not touch instrumentation.
    """
    out: list[str] = []

    counters: Mapping[str, Any] = snapshot.get("counters") or {}
    seen_help: set[str] = set()
    for key in sorted(counters):
        name, labels = _split_key(key)
        metric = f"{prefix}_{name}_total"
        if metric not in seen_help:
            out.append(f"# TYPE {metric} counter")
            seen_help.add(metric)
        out.append(f"{metric}{_labels_out(labels)} {int(counters[key])}")

    latency: Mapping[str, Any] = snapshot.get("latency_ms") or {}
    if latency:
        metric = f"{prefix}_latency_ms"
        out.append(f"# TYPE {metric} summary")
        out.append(f"# UNIT {metric} milliseconds")
        for name in sorted(latency):
            st = latency[name]
            base_name, labels = _split_key(name)
            for q in ("p50", "p95", "p99"):
                if st.get(q) is None:
                    continue
                quantile = {"p50": "0.5", "p95": "0.95", "p99": "0.99"}[q]
                out.append(
                    f"{metric}{_labels_out(labels, {'op': base_name, 'quantile': quantile})} "
                    f"{float(st[q])}"
                )
            out.append(
                f"{metric}_count{_labels_out(labels, {'op': base_name})} {int(st.get('n', 0))}"
            )
            if st.get("max") is not None:
                out.append(
                    f"{prefix}_latency_ms_max{_labels_out(labels, {'op': base_name})} "
                    f"{float(st['max'])}"
                )

    cal: Mapping[str, Any] = snapshot.get("calibration") or {}
    for field_name, metric_suffix, mtype in (
        ("ece", "calibration_ece", "gauge"),
        ("brier", "calibration_brier", "gauge"),
        ("n", "calibration_window_samples", "gauge"),
    ):
        val = cal.get(field_name)
        if val is None:
            continue
        metric = f"{prefix}_{metric_suffix}"
        out.append(f"# TYPE {metric} {mtype}")
        out.append(f"{metric} {float(val)}")
    if "alert" in cal:
        metric = f"{prefix}_calibration_alert"
        out.append(f"# TYPE {metric} gauge")
        out.append(f"{metric} {1 if cal.get('alert') else 0}")

    out.append("# EOF")
    return "\n".join(out) + "\n"


def _q(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]
