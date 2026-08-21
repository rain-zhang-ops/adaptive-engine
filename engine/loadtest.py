"""Capacity measurement -- QPS and tail latency under a mixed read/write load.

Why this exists as a script and not a paragraph in the README: capacity claims
that are not measured are made up, and the two numbers that matter here are not
guessable from the code.

* ``/v1/next`` is CPU-bound numpy over the recalled pool. It scales with cores.
* ``/v1/signals`` takes ``BEGIN IMMEDIATE``, so **writes serialise across the
  whole process**. Concurrency does not help it, and past some write share the
  read path starts queueing behind it. A read-only benchmark would hide exactly
  the limit an operator needs to know.

Two clocks are reported per endpoint and the gap between them is the point:

* **client wall time** -- what the caller experiences, including queueing
* **server-side engine time** -- from ``/metrics.json``, the decide/observe work itself

If client p99 is far above engine p99, the bottleneck is contention or the event
loop, not the engine; adding replicas helps. If they track each other, the engine
itself is the limit and the pool/recall budget is the lever.

Usage::

    python -m engine.loadtest                        # defaults, ~30s
    python -m engine.loadtest --concurrency 32 --duration 20 --write-share 0.3
    python -m engine.loadtest --url 127.0.0.1:8080 --api-key dev-key   # a real deployment

Numbers are only meaningful together with the hardware line printed in the header.
"""


from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import tempfile
import threading
import time
from collections import defaultdict
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

KEY = "loadtest-key"
TENANT = "loadtest"


# ---------------------------------------------------------------------------
# server under test
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_server(db_path: Path, port: int):
    """Run the real ASGI app under uvicorn in a thread.

    Deliberately not TestClient: TestClient bypasses the event loop and the
    socket, which are two of the three things being measured.
    """
    import uvicorn

    from engine.api import ApiKeyRegistry, RateLimiter, create_app
    from engine.store import SqliteStore

    keys = ApiKeyRegistry()
    keys.add(KEY, TENANT)
    app = create_app(
        store=SqliteStore(db_path),
        keys=keys,
        # limits are a policy question, not a capacity question -- lift them so
        # the measurement is of the engine and not of the token bucket
        limiter=RateLimiter(rate_per_sec=1e9, burst=1e9),
    )
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                         access_log=False)
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            return server, t
        time.sleep(0.05)
    raise RuntimeError("server did not start")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class Client:
    """One keep-alive connection per worker thread.

    A fresh connection per request would measure TCP setup, which no real client
    does and which would flatter or flatten the tail depending on the OS.
    """

    def __init__(self, host: str, port: int, key: str = KEY) -> None:
        self.con = HTTPConnection(host, port, timeout=30)
        self.hdr = {"X-API-Key": key, "Content-Type": "application/json"}

    def post(self, path: str, body: Any) -> tuple[int, Any]:
        payload = json.dumps(body)
        self.con.request("POST", path, payload, self.hdr)
        r = self.con.getresponse()
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)

    def get(self, path: str) -> tuple[int, Any]:
        self.con.request("GET", path, headers=self.hdr)
        r = self.con.getresponse()
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)

    def close(self) -> None:
        self.con.close()


def _seed(mk: Any, n_items: int) -> list[str]:
    from engine.simulator import SimConfig, Simulator

    sim = Simulator(SimConfig(n_items=n_items, n_users=1))
    c = mk()
    ids: list[str] = []
    batch: list[dict[str, Any]] = []
    for it in sim.items:
        batch.append({"id": it.id, "tags": dict(it.tag_weights),
                      "difficulty_prior": it.difficulty_prior,
                      "attrs": dict(it.attrs)})
        ids.append(it.id)
        if len(batch) == 500:
            st, _ = c.post("/v1/items", {"items": batch})
            assert st == 200, st
            batch = []
    if batch:
        st, _ = c.post("/v1/items", {"items": batch})
        assert st == 200, st
    c.close()
    return ids



# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


# A percentile needs enough samples to mean anything: with 44 samples the "p99"
# is just the maximum. Below these counts the quantile is reported as null so a
# reader does not mistake an order statistic near the tail for a real percentile.
_MIN_SAMPLES = {"p50": 20, "p95": 100, "p99": 100}


def _summarise(name: str, lat: list[float], errs: int, elapsed: float) -> dict[str, Any]:
    n = len(lat)

    def q(label: str, frac: float):
        return round(_pct(lat, frac), 2) if n >= _MIN_SAMPLES[label] else None

    return {
        "endpoint": name,
        "requests": n,
        "errors": errs,
        "qps": round(n / elapsed, 1) if elapsed > 0 else 0.0,
        "p50": q("p50", 0.50),
        "p95": q("p95", 0.95),
        "p99": q("p99", 0.99),
        "max": round(max(lat), 2) if lat else 0.0,
        "mean": round(statistics.fmean(lat), 2) if lat else 0.0,
    }



def run(concurrency: int, duration: float, n_items: int, n_users: int,
        count: int, write_share: float, warmup: float,
        target: str | None = None, api_key: str = KEY) -> dict[str, Any]:
    """Drive load against a server. With ``target`` (``host:port``) an already
    running instance is measured -- which is the only way to measure a
    multi-worker or multi-replica deployment, since the numbers below show that
    capacity comes from processes rather than threads."""
    tmp = None
    server = None
    if target:
        host, _, p = target.partition(":")
        host = host or "127.0.0.1"
        port = int(p or 80)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="loadtest-"))
        host, port = "127.0.0.1", _free_port()
        server, _thread = _start_server(tmp / "load.db", port)
    try:
        def mk() -> Client:
            return Client(host, port, api_key)

        item_ids = _seed(mk, n_items)

        lat: dict[str, list[float]] = defaultdict(list)
        errs: dict[str, int] = defaultdict(int)
        lock = threading.Lock()
        stop = threading.Event()

        def worker(wid: int) -> None:
            c = mk()
            mine: dict[str, list[float]] = defaultdict(list)
            mine_err: dict[str, int] = defaultdict(int)
            n = 0
            # deterministic per-worker interleaving; no shared RNG, no lock on
            # the hot path -- a contended lock would show up as our own tail
            try:
                while not stop.is_set():
                    n += 1
                    user = f"u{(wid * 7919 + n) % n_users}"
                    is_write = write_share > 0 and (n % max(1, int(round(1 / write_share))) == 0)
                    t0 = time.perf_counter()
                    if is_write:
                        j = (wid * 104729 + n) % len(item_ids)
                        st, _ = c.post("/v1/signals", {"signals": [{
                            "user": user, "item": item_ids[j],
                            "outcome": float(n % 2), "ts": time.time(),
                            "signal_id": f"lt-{wid}-{n}", "propensity": 1.0}]})
                        name = "signals"
                    else:
                        st, _ = c.post("/v1/next", {"user": user, "count": count})
                        name = "next"
                    ms = (time.perf_counter() - t0) * 1000.0
                    if st != 200:
                        mine_err[name] += 1
                    else:
                        mine[name].append(ms)
            finally:
                c.close()
                with lock:
                    for k, v in mine.items():
                        lat[k].extend(v)
                    for k, v in mine_err.items():
                        errs[k] += v

        # warmup is not cosmetic: first call pays import, catalogue read and
        # SQLite page cache warming, and would otherwise land in the tail
        warm = mk()
        t_end = time.time() + warmup
        while time.time() < t_end:
            warm.post("/v1/next", {"user": "warmup", "count": count})
        warm.close()

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(concurrency)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        time.sleep(duration)
        stop.set()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.perf_counter() - t0

        # Union of both key sets: an endpoint that returned non-200 on *every*
        # request never lands in `lat`, and keying the table off `lat` alone
        # would silently drop the one row a reader most needs to see.
        client_side = [_summarise(k, lat.get(k, []), errs.get(k, 0), elapsed)
                       for k in sorted(set(lat) | set(errs))]
        total = sum(len(v) for v in lat.values())

        c = mk()
        _, snap = c.get("/metrics.json")
        c.close()

        return {
            "config": {"concurrency": concurrency, "duration_s": round(elapsed, 2),
                       "items": n_items, "users": n_users, "count": count,
                       "write_share": write_share, "cpu_count": os.cpu_count(),
                       "target": target or "in-process (single worker)"},
            "client_side": client_side,
            "total_qps": round(total / elapsed, 1) if elapsed else 0.0,
            "server_side_engine_ms": (snap or {}).get("latency_ms", {}),
            "server_counters": (snap or {}).get("counters", {}),
            # The server-side window spans the whole process, so it still includes
            # the warmup requests' cold-path samples (import, catalogue read, cold
            # page cache) that the client-side numbers exclude. The two clocks are
            # therefore not directly subtractable at the tail; treat the gap as
            # indicative, not exact. Under --url with N workers this is also one
            # worker's local view, not an aggregate.
            "server_side_caveat": ("includes warmup cold-path; per-process "
                                   "(one worker under --url)"),
        }

    finally:
        if server is not None:
            server.should_exit = True
            time.sleep(0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--items", type=int, default=3000)
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--write-share", type=float, default=0.2,
                    help="fraction of requests that are /v1/signals writes")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--url", default=None, metavar="HOST:PORT",
                    help="measure an already-running server (e.g. uvicorn with "
                         "--workers N) instead of starting a single-worker one")
    ap.add_argument("--api-key", default=KEY,
                    help="API key for --url; ignored for the in-process server")
    args = ap.parse_args()

    res = run(args.concurrency, args.duration, args.items, args.users,
              args.count, args.write_share, args.warmup,
              target=args.url, api_key=args.api_key)

    cfg = res["config"]
    print("=" * 72)
    print("LOAD TEST -- adaptive-engine")
    print(f"  target={cfg['target']}")
    print(f"  concurrency={cfg['concurrency']}  duration={cfg['duration_s']}s  "
          f"cpu_count={cfg['cpu_count']}")
    print(f"  catalogue={cfg['items']} items  users={cfg['users']}  "
          f"count={cfg['count']}  write_share={cfg['write_share']}")
    print("=" * 72)

    def cell(v: Any) -> str:
        # percentiles are None when the sample is too small to support them
        text = "n/a" if v is None else str(v)
        return f"{text:>8}"

    print(f"\ntotal throughput: {res['total_qps']} req/s\n")
    print("client-side wall latency (ms)")
    print(f"  {'endpoint':10} {'reqs':>7} {'errs':>5} {'qps':>8} "
          f"{'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")

    for row in res["client_side"]:
        print(f"  {row['endpoint']:10} {row['requests']:>7} {row['errors']:>5} "
              f"{cell(row['qps'])} {cell(row['p50'])} {cell(row['p95'])} "
              f"{cell(row['p99'])} {cell(row['max'])}")

    print("\nserver-side engine latency (ms, from /metrics.json)")
    for name, st in sorted(res["server_side_engine_ms"].items()):
        print(f"  {name:10} n={st['n']:>7} p50={st['p50']:>8} "
              f"p95={st['p95']:>8} p99={st['p99']:>8} max={st['max']:>8}")

    print("\ninterpretation: client p99 >> engine p99 means queueing/contention "
          "(add workers/replicas);\n  the two tracking each other means the engine "
          "itself is the ceiling (tune recall_limit / count).")
    print("\nNOTE: a decide is CPU-bound Python+numpy, so threads inside one "
          "process do not add\n  capacity -- measured throughput *falls* as "
          "in-process concurrency rises. Capacity comes\n  from processes "
          "(uvicorn --workers N) or replicas, measured via --url; cap the "
          "threadpool\n  with ADAPTIVE_MAX_CONCURRENCY=2 so the excess queues "
          "instead of thrashing.")
    print("NOTE: writes take BEGIN IMMEDIATE and therefore serialise per "
          "process.\n  Re-run with --write-share 0 to see the read-only ceiling.")



if __name__ == "__main__":
    main()
