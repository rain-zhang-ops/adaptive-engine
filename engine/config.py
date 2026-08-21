"""Runtime configuration: one resolved answer for how this process should run.

Why a file at all, when everything here already had an environment variable: the
knobs that decide *capacity* are no longer one-liners. Pool size trades decision
quality for latency, the threadpool cap trades tail for throughput, and the worker
count interacts with both -- a deployment picks a coherent set of them, and a set
belongs in something reviewable, diffable and shippable rather than in a shell
line that has to be reconstructed from a container's history.

Precedence is **file over environment**, with defaults underneath. The file is the
declared intent; an environment variable is what a shell happened to be holding.
Reversing that would mean a stray `ADAPTIVE_DB` in an operator's profile silently
beating a checked-in configuration, and nothing in the file would be trustworthy.

Constructor arguments beat both. ``create_app(store=...)`` is code, and code that
hands in a store is not asking to be told which database to open.

Not read here: ``ADAPTIVE_API_KEYS`` and the admin/metrics tokens. Secrets stay in
the environment, because a config file is the thing most likely to be committed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["RuntimeConfig", "ConfigError", "load_runtime", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = Path("adaptive.yaml")
"""Looked for in the working directory when ``ADAPTIVE_CONFIG`` is unset. Absent is
not an error -- a zero-config start is the point of the defaults below."""

MEMORY_DB = ":memory:"


class ConfigError(RuntimeError):
    """Raised at startup, never at request time. A configuration that cannot be
    understood must stop the process: serving on a guessed value is how a cluster
    ends up half-configured with nothing in the logs to say so."""


@dataclass(frozen=True)
class RuntimeConfig:
    # -- storage
    db: str = "adaptive.db"

    # -- capacity
    workers: int = 1
    max_concurrency: int | None = None
    """Starlette threadpool ceiling. ``None`` leaves its 40-thread default, which is
    sized for blocking I/O and measured 2.5x worse than 2 for this workload."""

    # -- rate limiting (per process, so N workers multiply it)
    rate_per_sec: float = 50.0
    burst: float = 200.0

    # -- candidate generation
    recall_limit: int = 800
    recall_tags: int = 12
    explore_pool_factor: int = 3

    notes: tuple[str, ...] = ()
    """Adjustments made while resolving, e.g. a worker count forced down. Surfaced
    rather than applied silently, because "why is it running one worker" must be
    answerable from the logs."""

    @property
    def in_memory(self) -> bool:
        return self.db == MEMORY_DB


# Config key -> the environment variable it also accepts. Keys absent here are
# file-only: they are new, and inventing a second surface for them by default
# would double the number of places a future reader has to check.
_ENV_OF = {
    "db": "ADAPTIVE_DB",
    "workers": "WORKERS",
    "max_concurrency": "ADAPTIVE_MAX_CONCURRENCY",
    "rate_per_sec": "ADAPTIVE_RATE_PER_SEC",
    "burst": "ADAPTIVE_BURST",
}

_FIELDS = ("db", "workers", "max_concurrency", "rate_per_sec", "burst",
           "recall_limit", "recall_tags", "explore_pool_factor")


def _as_int(key: str, raw: Any, minimum: int) -> int:
    try:
        v = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{key}={raw!r} is not an integer") from None
    if v < minimum:
        raise ConfigError(f"{key}={raw!r} must be >= {minimum}")
    return v


def _as_float(key: str, raw: Any) -> float:
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{key}={raw!r} is not a number") from None
    if v <= 0.0 or v != v:
        raise ConfigError(f"{key}={raw!r} must be > 0")
    return v


def _coerce(key: str, raw: Any) -> Any:
    if key == "db":
        text = str(raw).strip()
        if not text:
            raise ConfigError("db must not be empty")
        return text
    if key == "max_concurrency":
        # Explicit null means "leave Starlette's default", which is not the same
        # as 0 -- 0 would be a process that accepts requests and runs none.
        return None if raw is None else _as_int(key, raw, 1)
    if key in ("rate_per_sec", "burst"):
        return _as_float(key, raw)
    return _as_int(key, raw, 1)


def _read_file(path: Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        env = os.environ.get("ADAPTIVE_CONFIG")
        if env and env.strip():
            path = Path(env.strip())
        elif DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
        else:
            return {}, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except FileNotFoundError:
        # An explicitly named file that does not exist is a mistake, unlike the
        # default path being absent: someone asked for this file by name.
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    if doc is None:
        return {}, path
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{path} must contain a mapping, got {type(doc).__name__}")
    unknown = [k for k in doc if k not in _FIELDS]
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}; known keys are {list(_FIELDS)}")
    return dict(doc), path


def load_runtime(path: str | Path | None = None) -> RuntimeConfig:
    """Resolve file, then environment, then defaults, into one frozen answer."""
    doc, source = _read_file(Path(path) if path is not None else None)

    values: dict[str, Any] = {}
    for key in _FIELDS:
        if key in doc:
            values[key] = _coerce(key, doc[key])
            continue
        env_var = _ENV_OF.get(key)
        raw = os.environ.get(env_var) if env_var else None
        if raw is not None and raw.strip():
            values[key] = _coerce(key, raw)

    cfg = RuntimeConfig(**values)

    notes: list[str] = []
    if source is not None:
        notes.append(f"config file: {source}")
    if cfg.in_memory and cfg.workers > 1:
        # Each worker would open its own private empty database, so one tenant's
        # data would be split across processes and two requests from the same user
        # could be answered from different histories. Not a tuning question.
        notes.append(f"workers forced to 1: an in-memory database cannot be shared "
                     f"across processes (requested {cfg.workers})")
        cfg = RuntimeConfig(**{**values, "workers": 1})
    if cfg.in_memory:
        notes.append("in-memory database: all learned state is lost on restart")

    return RuntimeConfig(**{f: getattr(cfg, f) for f in _FIELDS}, notes=tuple(notes))


def main(argv: list[str] | None = None) -> int:
    """``python -m engine.config [key]`` -- the resolved value, for entrypoints.

    The container's CMD needs the worker count *after* resolution, since the file
    may set it and an in-memory database may cap it. Printing it here keeps that
    logic in one place instead of duplicating the precedence rules in shell.
    """
    args = sys.argv[1:] if argv is None else argv
    try:
        cfg = load_runtime()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    for note in cfg.notes:
        print(f"# {note}", file=sys.stderr)
    if not args:
        for f in _FIELDS:
            print(f"{f}: {getattr(cfg, f)}")
        return 0
    key = args[0]
    if key not in _FIELDS:
        print(f"unknown key {key!r}; known keys are {list(_FIELDS)}", file=sys.stderr)
        return 2
    print(getattr(cfg, key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
