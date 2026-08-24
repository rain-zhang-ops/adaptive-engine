"""HTTP layer -- the transport translation of ``openapi.yaml``.

Thin on purpose. Its whole job is: authenticate, resolve the tenant, translate
the external vocabulary into an ``EngineService`` call, and render the result
back. No decision logic lives here, because anything that lives here is
unreachable from the library API and therefore untestable without a server.

What this layer owns that the engine deliberately does not:

*Identity.* An API key resolves to a tenant. The engine never guesses a tenant
from a payload field -- if it did, one client could read another's state by
sending a different string. Keys are stored hashed, and looked up in the database
so that rotation and revocation take effect without a restart.

*Rate limiting.* A token bucket per tenant. Being explicit about the limitation:
the bucket is in-process, so with N replicas the effective limit is N times the
configured one. That is fine for a private deployment and wrong for multi-tenant
SaaS at scale, where the counter has to be shared.

*Liveness vs readiness.* ``/healthz`` answers "this process is running" and must
not touch the database -- otherwise a slow query takes the pod down. ``/readyz``
answers "this process can serve", which requires actually consulting the
dependency. A single endpoint that returns ``ok`` without checking anything
reports health it has not verified.

*Human-readable ``why``.* The kernel emits machine-readable contributions and
has no vocabulary to render into -- that is the point of the iron rule. Turning
them into a sentence is a presentation concern and belongs here, where a tenant
can override the templates without touching the engine.

Error philosophy: 200 with ``confidence: low`` and a ``fallback_reason`` for
anything the engine can degrade through; 4xx only for a request that is
genuinely malformed or unauthorised. A 5xx from a recommendation service turns
its own hiccup into the caller's outage.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping


import anyio.to_thread
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, model_validator


from contracts.core import Item
from engine.config import load_runtime
from engine.observability import Metrics, Timer, log_event, render_openmetrics
from engine.policy import PolicyError, load_catalog
from engine.predicates import PATH_SYNTAX, PredicateError, is_valid_path
from engine.rendering import Renderer, load_renderer
from engine.service import EngineService, ServiceConfig
from engine.store import SqliteStore


__all__ = ["create_app", "ApiKeyRegistry", "RateLimiter", "render_why"]



# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


class ApiKeyRegistry:
    """API key -> tenant. Keys are held as SHA-256 digests, never in the clear.

    Two sources, in this order:

    1. an in-memory map, bootstrapped from the environment -- so a fresh
       deployment has a way in before any database row exists;
    2. the ``api_keys`` table -- so keys can be added, expired and revoked while
       the process is running. An env-only design means every rotation is a
       restart, which in practice means rotation does not happen.

    Lookups are cached for a few seconds. Without it every request costs a query;
    with an unbounded cache a revocation would not take effect. The window is the
    revocation lag and is stated rather than hidden.

    The cache is size-capped and negative results get a shorter TTL of their own.
    Caching misses forever, keyed by digest, means an unauthenticated caller
    sending random keys grows the process's memory one permanent entry at a time.
    """

    MAX_CACHE = 4096

    def __init__(self, store: SqliteStore | None = None, cache_ttl: float = 5.0,
                 miss_ttl: float = 1.0) -> None:
        self.store = store
        self.cache_ttl = cache_ttl
        self.miss_ttl = miss_ttl
        self._static: dict[str, str] = {}
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._lock = threading.Lock()


    @staticmethod
    def digest(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def add(self, key: str, tenant: str) -> None:
        self._static[self.digest(key)] = tenant

    def issue(self, key: str, tenant: str, label: str | None = None,
              expires_at: float | None = None) -> str:
        """Persist a key so it survives restarts and can later be revoked."""
        if self.store is None:
            raise RuntimeError("no store bound; cannot persist keys")
        d = self.digest(key)
        self.store.add_api_key(d, tenant, label=label, expires_at=expires_at)
        self._invalidate(d)
        return d

    def revoke(self, key: str | None = None, digest: str | None = None) -> bool:
        if self.store is None:
            raise RuntimeError("no store bound; cannot revoke keys")
        d = digest or (self.digest(key) if key else None)
        if not d:
            return False
        ok = self.store.revoke_api_key(d)
        self._static.pop(d, None)
        self._invalidate(d)
        return ok

    def _invalidate(self, digest: str) -> None:
        with self._lock:
            self._cache.pop(digest, None)

    def resolve(self, key: str | None, now: float | None = None) -> str | None:
        if not key:
            return None
        d = self.digest(key)
        t = time.time() if now is None else now

        with self._lock:
            hit = self._cache.get(d)
            if hit is not None and hit[1] > t:
                return hit[0]

        tenant = self._static.get(d)
        if tenant is None and self.store is not None:
            tenant = self.store.tenant_for_key(d, now=t)

        with self._lock:
            ttl = self.cache_ttl if tenant is not None else self.miss_ttl
            if len(self._cache) >= self.MAX_CACHE:
                self._evict(t)
            self._cache[d] = (tenant, t + ttl)
        return tenant

    def _evict(self, now: float) -> None:
        """Drop expired entries; if that frees nothing, drop the whole cache.

        Called under ``self._lock``. Clearing outright costs one extra query per
        live key rather than letting the map grow without bound, which is the
        cheaper failure of the two.
        """
        stale = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for k in stale:
            del self._cache[k]
        if not stale:
            self._cache.clear()


    @classmethod
    def from_env(cls, store: SqliteStore | None = None,
                 var: str = "ADAPTIVE_API_KEYS") -> "ApiKeyRegistry":
        """``ADAPTIVE_API_KEYS="key1:tenantA,key2:tenantB"``.

        Environment rather than a config file so keys never land in the repo.
        """
        reg = cls(store=store)
        for pair in (os.environ.get(var) or "").split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            key, tenant = pair.split(":", 1)
            reg.add(key.strip(), tenant.strip())
        return reg


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """Token bucket per tenant.

    A bucket allows short bursts while bounding the sustained rate, which is what
    a batch-oriented client actually needs -- a flat per-second cap would reject
    legitimate bulk ingestion.

    Callers pass ``cost``, because charging one token per request prices a
    500-user batch the same as ``/healthz``. Idle buckets are swept so the state
    map does not grow one permanent entry per key that ever appeared.
    """

    rate_per_sec: float = 50.0
    burst: float = 200.0
    idle_ttl: float = 600.0
    max_buckets: int = 8192
    _state: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow(self, tenant: str, cost: float = 1.0, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        with self._lock:
            if len(self._state) >= self.max_buckets:
                self._sweep(t)
            tokens, last = self._state.get(tenant, (self.burst, t))
            tokens = min(self.burst, tokens + (t - last) * self.rate_per_sec)
            if tokens < cost:
                self._state[tenant] = (tokens, t)
                return False
            self._state[tenant] = (tokens - cost, t)
            return True

    def _sweep(self, now: float) -> None:
        """Drop buckets idle long enough to have refilled anyway.

        Called under ``self._lock``. A bucket untouched for ``idle_ttl`` is
        indistinguishable from a fresh one, so forgetting it changes no decision.
        """
        for k in [k for k, (_, last) in self._state.items()
                  if now - last > self.idle_ttl]:
            del self._state[k]
        if len(self._state) >= self.max_buckets:
            self._state.clear()

    @classmethod
    def from_env(cls) -> "RateLimiter":
        """Resolve the limits through ``engine.config`` (file over environment)."""
        rc = load_runtime()
        return cls(rate_per_sec=rc.rate_per_sec, burst=rc.burst)



# ---------------------------------------------------------------------------
# wire models
# ---------------------------------------------------------------------------


class TuneIn(BaseModel):
    """Adjective knobs.

    Typed as plain strings rather than ``Literal`` on purpose: the value set lives
    in goals.yaml, and letting pydantic reject it here would produce a schema
    traceback instead of the policy layer's "did you mean 'moderate'?". One place
    owns the vocabulary, and it is the one that can suggest a near miss.

    Unknown keys are passed through for the same reason. Pydantic's default is to
    drop them silently, which is the worst of the three options -- a caller who
    typed ``freshnes`` would get a 200 and no freshness. Forwarded, they reach the
    layer that can say what was meant.
    """

    model_config = {"extra": "allow"}

    difficulty: str | None = None
    focus: str | None = None
    freshness: float | None = Field(default=None, ge=0.0, le=1.0)
    stakes: str | None = None

    def as_tune(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class WithinIn(BaseModel):
    """Scope the candidate set. ``additionalProperties: false`` in the contract,
    so unknown keys are rejected rather than dropped -- ``item_id`` for
    ``item_ids`` would otherwise return 200 having scoped nothing."""

    model_config = {"extra": "forbid"}

    tags: list[str] | None = None
    attrs: dict[str, Any] | None = None
    item_ids: list[str] | None = None


class ExcludeIn(BaseModel):
    model_config = {"extra": "forbid"}

    item_ids: list[str] | None = None
    attrs: dict[str, Any] | None = None
    max_per_tag: int | None = Field(default=None, ge=1)


class QuotaIn(BaseModel):
    model_config = {"extra": "forbid"}

    group_by: str
    counts: dict[str, int]



class NextRequest(BaseModel):
    user: str | None = None
    users: list[str] | None = Field(default=None, max_length=500)
    count: int = Field(default=10, ge=1, le=200)
    goal: str | None = None
    policy_ref: str | None = None
    """L3: a policy this tenant registered via ``POST /v1/policies``. Mutually
    exclusive with ``goal`` -- accepting both would leave it ambiguous which one
    produced the decision that gets audited."""

    tune: TuneIn | None = None
    within: WithinIn | None = None
    exclude: ExcludeIn | None = None
    quota: list[QuotaIn] | None = None
    explain: bool = True
    locale: str | None = None
    """Language for ``why`` and ``hint``. Falls back to the server default, then
    to the default locale in why.yaml."""

    @model_validator(mode="after")
    def _one_of(self):
        if bool(self.user) == bool(self.users):
            raise ValueError("provide exactly one of 'user' or 'users'")
        if self.goal and self.policy_ref:
            raise ValueError("provide either 'goal' or 'policy_ref', not both")
        return self

    def user_list(self) -> list[str]:
        return [self.user] if self.user else list(self.users or [])


class SignalIn(BaseModel):
    """One reported outcome.

    ``additionalProperties: false`` in the contract, and enforced here: a
    misspelt ``propensty`` silently dropped is the worst of the three options,
    because the field it drops is the one that cannot be reconstructed later.
    """

    model_config = {"extra": "forbid"}

    user: str
    item: str
    outcome: float = Field(ge=0.0, le=1.0)

    ts: float | None = None
    """Event time, epoch seconds. Optional -- omitted means "now", because the
    common case is reporting something that just happened and forcing every
    caller to synthesise a timestamp only invites unit mistakes."""

    signal_id: str | None = None
    """Idempotency key. Without one a redelivery is counted twice, so its absence
    is reported back in ``warnings`` rather than left to be discovered later."""

    decision_id: str | None = None
    """The decision this outcome answers. Supplying it lets the server look the
    propensity up, so the caller does not have to keep a per-item propensity map
    of its own."""

    propensity: float | None = Field(default=None, gt=0.0, le=1.0)
    policy_id: str | None = None
    context: dict[str, Any] | None = None


class SignalsRequest(BaseModel):
    signals: list[SignalIn] = Field(min_length=1, max_length=5000)


class ItemIn(BaseModel):
    id: str
    tags: dict[str, float] | None = None
    difficulty_prior: float | None = Field(default=None, ge=0.0, le=1.0)
    attrs: dict[str, Any] | None = None


class ItemsRequest(BaseModel):
    items: list[ItemIn] = Field(min_length=1, max_length=5000)


class PolicyIn(BaseModel):
    """An L3 policy document.

    Deliberately not mirrored field-by-field into pydantic: the authority is the
    engine's own load-time validation in ``engine/policy.py``, and a second
    partial copy of that shape here would drift. ``contracts/policy.schema.json``
    documents the same surface but is not executed by anything, so it is a
    reference rather than a gate. Unknown keys are kept so the policy layer can
    reject them with a reason.
    """

    model_config = {"extra": "allow"}
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    label: str | None = None


class PurgeRequest(BaseModel):
    model_config = {"extra": "forbid"}

    prediction_ttl_days: float = Field(default=30.0, ge=0.0)
    signal_ttl_days: float = Field(default=400.0, ge=0.0)
    decision_ttl_days: float = Field(default=90.0, ge=0.0)
    dry_run: bool = False
    """Count what would be deleted and delete nothing. Purge is irreversible and
    takes the audit trail with it, so the blast radius is inspectable first."""




# ---------------------------------------------------------------------------
# why rendering
# ---------------------------------------------------------------------------
#
# The actual renderer lives in engine.rendering and is driven by
# contracts/why.yaml, so templates and locales are data. This free function is a
# thin default kept for the library API and existing callers; the HTTP layer uses
# the app's own Renderer so a tenant can supply its own why.yaml.

_DEFAULT_RENDERER: Renderer | None = None


def _default_renderer() -> Renderer:
    global _DEFAULT_RENDERER
    if _DEFAULT_RENDERER is None:
        _DEFAULT_RENDERER = load_renderer()
    return _DEFAULT_RENDERER


def render_why(chosen_reasons: Mapping[str, float], p_hat: float,
               locale: str | None = None) -> str:
    """Machine-readable contributions -> one sentence, via the default renderer."""
    return _default_renderer().why(chosen_reasons, p_hat, locale)



# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def _constraints_spec(req: NextRequest) -> dict[str, Any]:
    """Translate the friendly external shape into kernel constraints.

    Attribute filters become predicates, which the engine compiles with its own
    safe grammar -- the strings never reach ``eval``.

    Keys are checked against that grammar *here*, while the request is still in
    hand. Interpolating an unchecked key produced an expression that only failed
    when the chooser compiled it mid-request, which surfaced as a 5xx for an
    ordinary attribute name such as ``item-kind``. A key the grammar cannot
    address is a caller error, so it is a 400 naming the key.
    """
    spec: dict[str, Any] = {}
    preds: list[str] = []
    if req.within and req.within.attrs:
        for k, v in req.within.attrs.items():
            _check_attr_key(k, "within.attrs")
            preds.append(f"attrs.{k} == {_lit(v)}")
    if req.exclude and req.exclude.attrs:
        for k, v in req.exclude.attrs.items():
            _check_attr_key(k, "exclude.attrs")
            preds.append(f"attrs.{k} != {_lit(v)}")
    if preds:
        spec["predicates"] = preds
    if req.within and req.within.tags:
        spec["within_tags"] = list(req.within.tags)
    if req.exclude and req.exclude.item_ids:
        spec["exclude_item_ids"] = req.exclude.item_ids
    if req.exclude and req.exclude.max_per_tag:
        spec["max_per_tag"] = req.exclude.max_per_tag
    if req.quota:
        spec["quotas"] = [{"group_by": q.group_by, "counts": q.counts} for q in req.quota]
    return spec


def _check_attr_key(key: str, where: str) -> None:
    if not is_valid_path(key):
        raise PolicyError(
            f"{where}: {key!r} is not an addressable attribute name "
            f"({PATH_SYNTAX})")


def _lit(v: Any) -> str:
    """Render a JSON value as a predicate literal.

    ``None`` and lists get their own forms rather than being stringified: an
    ``attrs.k == 'None'`` or ``attrs.k == '[a, b]'` compiles fine and then never
    matches anything, which is a filter that fails without saying so.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            raise PolicyError(
                f"non-finite number {v!r} cannot be used as an attribute filter")
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    if isinstance(v, Mapping):
        raise PolicyError("attribute filters compare scalars or lists, not objects")
    return "'" + str(v).replace("'", "") + "'"


_STATUS_NAMES = {400: "bad_request", 401: "unauthorized", 403: "forbidden",
                 404: "not_found", 409: "conflict", 422: "validation",
                 429: "rate_limited"}


def _error(status: int, code: str, detail: str) -> HTTPException:
    """A 4xx in the same envelope as every other 4xx.

    ``openapi.yaml`` types every error response as ``{error, detail}``. A bare
    ``HTTPException`` renders FastAPI's default ``{detail}``, so a generated
    client fails schema validation on exactly the responses it must handle.
    """
    return HTTPException(status_code=status, detail={"error": code, "detail": detail})


def _origin_of(request: Request) -> str:
    client = request.client
    return client.host if client and client.host else "unknown"


def _bearer_matches(header: str | None, token: str) -> bool:
    """Constant-time bearer comparison that cannot raise.

    ``hmac.compare_digest`` rejects non-ASCII ``str`` with ``TypeError``, and
    Starlette decodes headers as latin-1 -- so ``Authorization: Bearer e``` with
    an accent would 500 from inside the auth check. Compare bytes instead.
    """
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:].encode("utf-8", "surrogateescape"),
                               token.encode("utf-8"))




def create_app(
    store: SqliteStore | None = None,
    keys: ApiKeyRegistry | None = None,
    limiter: RateLimiter | None = None,
    cfg: ServiceConfig | None = None,
    metrics_token: str | None = None,
    renderer: Renderer | None = None,
) -> FastAPI:
    runtime = load_runtime()
    for note in runtime.notes:
        log_event("config", note=note)
    store = store or SqliteStore(runtime.db)
    keys = keys or ApiKeyRegistry.from_env(store=store)
    if keys.store is None:
        keys.store = store           # so DB-backed rotation works even if injected
    limiter = limiter or RateLimiter(rate_per_sec=runtime.rate_per_sec,
                                     burst=runtime.burst)
    # Failed auth never reaches the tenant bucket, so it needs its own. Without
    # it, key guessing is unthrottled: the 401 is raised before any accounting.
    auth_limiter = RateLimiter(rate_per_sec=5.0, burst=50.0)
    metrics = Metrics()

    catalog = load_catalog()          # fails at startup on a bad policy, by design
    # An injected cfg is code asking for exact values, so it is not second-guessed
    # by a file; without one, candidate generation follows the runtime config.
    cfg = cfg or ServiceConfig(recall_limit=runtime.recall_limit,
                               recall_tags=runtime.recall_tags,
                               explore_pool_factor=runtime.explore_pool_factor)
    svc = EngineService(store, catalog=catalog, cfg=cfg, metrics=metrics)
    scrape_token = metrics_token if metrics_token is not None else \
        os.environ.get("ADAPTIVE_METRICS_TOKEN")
    # Destructive operations get their own credential when one is configured.
    # Unset keeps single-tenant bootstrap working; set separates "can read and
    # report" from "can delete the audit trail".
    admin_token = os.environ.get("ADAPTIVE_ADMIN_TOKEN")

    render = renderer or load_renderer(os.environ.get("ADAPTIVE_WHY_FILE") or None)
    server_locale = os.environ.get("ADAPTIVE_LOCALE")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Bound how many decides run at once inside one process.

        Endpoints are sync ``def``, so Starlette hands each one to its threadpool
        -- 40 threads by default. That default is sized for blocking I/O, and a
        decide is neither: it is CPU-bound Python and numpy interleaved with
        GIL-releasing SQLite calls. Admitting 40 of them concurrently does not add
        throughput (there is one GIL), it only multiplies the handoff traffic
        around every SQLite call, and measured throughput *falls* past ~2
        concurrent decides while the tail grows without bound.

        A cap converts that into queueing: the same work, admitted in a width the
        process can actually execute, with excess waiting in the accept queue
        rather than thrashing inside it. Unset leaves Starlette's default so this
        is opt-in and nothing changes for an existing deployment.
        """
        if runtime.max_concurrency is not None:
            anyio.to_thread.current_default_thread_limiter().total_tokens = \
                runtime.max_concurrency
        try:
            yield
        finally:
            # Connections are per thread and tracked in the store precisely so
            # they can be released here; without this, shutdown leaks every
            # connection any worker thread ever opened.
            store.close()

    app = FastAPI(title="Adaptive Decision API", version="1.0.0", lifespan=lifespan)
    app.state.service = svc
    app.state.keys = keys
    app.state.limiter = limiter
    app.state.metrics = metrics
    app.state.renderer = render

    def tenant_of(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        t = request.app.state.keys.resolve(x_api_key)
        if t is None:
            metrics.incr("auth_rejected")
            # Throttled by origin, since there is no tenant to charge yet and an
            # unthrottled 401 path is a free key-guessing oracle.
            if not auth_limiter.allow(_origin_of(request)):
                metrics.incr("auth_throttled")
                raise _error(429, "rate_limited", "too many failed authentications")
            raise _error(401, "unauthorized", "invalid or missing X-API-Key")
        if not request.app.state.limiter.allow(t):
            metrics.incr("rate_limited", tenant=t)
            raise _error(429, "rate_limited", "rate limit exceeded")
        return t

    Tenant = Depends(tenant_of)

    def charge(tenant: str, cost: float) -> None:
        """Charge a request's remaining cost once its size is known.

        ``tenant_of`` charges one token for every request, which prices a
        500-user batch the same as ``/healthz``. Work here is roughly linear in
        users x count, so the rest is charged before any of it is done.
        """
        if cost <= 1.0:
            return
        if not limiter.allow(tenant, cost=cost - 1.0):
            metrics.incr("rate_limited", tenant=tenant)
            raise _error(429, "rate_limited",
                         "rate limit exceeded for a request of this size")


    @app.exception_handler(PolicyError)
    async def _policy_error(_: Request, exc: PolicyError):
        # A bad goal/tune is the caller's request error, not a server fault.
        return JSONResponse(status_code=400, content={"error": "policy", "detail": str(exc)})

    @app.exception_handler(PredicateError)
    async def _predicate_error(_: Request, exc: PredicateError):
        """A filter expression the grammar cannot compile is a request error.

        Compilation happens per request, and this class is a ``ValueError`` rather
        than a ``PolicyError``, so without this handler an unaddressable attribute
        name became a 5xx from a service whose contract says it never emits one.
        """
        return JSONResponse(status_code=400,
                            content={"error": "predicate", "detail": str(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        """One envelope for every 4xx, including the ones raised as plain
        ``HTTPException`` (401/404/429). FastAPI's default body omits ``error``,
        which ``openapi.yaml`` marks required."""
        detail = exc.detail
        if isinstance(detail, Mapping) and "error" in detail:
            content = dict(detail)
        else:
            content = {"error": _STATUS_NAMES.get(exc.status_code, "error"),
                       "detail": detail if isinstance(detail, str) else str(detail)}
        return JSONResponse(status_code=exc.status_code, content=content,
                            headers=getattr(exc, "headers", None))


    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        """Schema rejections in the same envelope as policy rejections.

        FastAPI's default 422 body is a list of pydantic internals: useful to
        whoever wrote the model, opaque to whoever is integrating. Same shape as
        every other 4xx here, one readable line per problem, and the raw detail
        kept alongside for anyone who wants it.
        """
        problems = []
        for e in exc.errors():
            where = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
            problems.append(f"{where or 'body'}: {e.get('msg', 'invalid')}")
        return JSONResponse(status_code=422, content={
            "error": "validation",
            "detail": "; ".join(problems),
            "problems": problems,
        })

    def _locale(req_locale: str | None,
                accept_language: str | None = None) -> str | None:
        return req_locale or server_locale or accept_language

    # -- decide ------------------------------------------------------------

    @app.post("/v1/next")
    def next_items(req: NextRequest, tenant: str = Tenant,
                   accept_language: str | None = Header(default=None)):
        timer = Timer()
        now = time.time()
        users = req.user_list()
        # Cost is charged before the work, and scaled by it: recall + scoring +
        # greedy selection runs once per user over a pool sized by count.
        charge(tenant, max(1.0, len(users) * (1.0 + req.count / 10.0)))
        spec = _constraints_spec(req)
        cand = req.within.item_ids if (req.within and req.within.item_ids) else None
        loc = _locale(req.locale, accept_language)

        results = svc.decide_many(tenant, users, count=req.count, goal=req.goal,
                                  tune=req.tune.as_tune() if req.tune else None,
                                  constraints_spec=spec, candidate_ids=cand, now=now,
                                  policy_ref=req.policy_ref)

        worst = "high"
        reasons: list[str] = []
        payload = []
        for uid, r in zip(users, results):
            worst = _min_conf(worst, r.decision.confidence)
            if r.decision.fallback_reason:
                reasons.append(r.decision.fallback_reason)
            payload.append({
                "user": uid,
                # The handle for this decision. Reporting outcomes with it is
                # enough -- the server already holds each item's propensity, so
                # the caller does not have to keep a copy of the one number that
                # cannot be reconstructed later.
                "decision_id": r.decision_id,
                # Per user, not just per batch. A batch-wide scalar cannot say
                # *who* degraded, and picking the first reason found while
                # reporting the worst confidence describes two different users.
                "confidence": r.decision.confidence,
                "fallback_reason": r.decision.fallback_reason,
                "hint": (render.hint(r.decision.fallback_reason, loc)
                         if r.decision.confidence != "high" else None),
                "recall": dict(r.recall),
                "items": [
                    {
                        "id": c.item_id,
                        "why": render.why(c.reasons, c.p_hat, loc),
                        "expected_success": round(c.p_hat, 4),
                        # Not rounded, unlike the cosmetic fields around it: this
                        # is the IPS denominator, and /v1/decisions/{id} replays
                        # the stored full-precision value. Rounding here made the
                        # served propensity disagree with the replayed one, so a
                        # caller mixing the two sources weighted the same
                        # impression two different ways.
                        "propensity": c.propensity,
                        **({"detail": {k: round(v, 6) for k, v in c.reasons.items()}}
                           if req.explain else {}),
                    }
                    for c in r.decision.chosen
                ],
            })

        # Batch-level summary stays for the single-user case, which is the common
        # one; for a batch it is the worst confidence and every distinct reason,
        # not one arbitrary reason standing in for the rest.
        reason = reasons[0] if reasons else None
        distinct = sorted(set(reasons))
        head = results[0] if results else None
        log_event("next", tenant=tenant, users=len(users), confidence=worst,
                  fallback=reason, degraded=len(reasons),
                  latency_ms=round(timer.ms, 2))
        return {
            "confidence": worst,
            "fallback_reason": reason,
            # What to do about a degraded result. Returning the reason without a
            # next step leaves the caller guessing whether to retry, wait, or
            # change the request.
            "hint": render.hint(reason, loc) if worst != "high" else None,
            "results": payload,
            "meta": {
                "policy_id": head.decision.policy_id if head else "",
                "model_version": head.decision.model_version if head else "",
                "goal": head.goal if head else None,
                "policy_ref": req.policy_ref,
                "applied_tune": dict(head.applied) if head else {},
                "notes": list(head.notes) if head else [],
                "warnings": list(head.warnings) if head else [],
                # Surfaced so "why wasn't item X considered?" is answerable, and so
                # a truncated catalogue is visible rather than silent. Per-user
                # recall lives on each result; this is the first user's.
                "recall": dict(head.recall) if head else {},
                "degraded_users": len(reasons),
                "fallback_reasons": distinct,
                "latency_ms": int(timer.ms),
            },
        }

    @app.get("/v1/decisions/{decision_id}")
    def get_decision(decision_id: str, tenant: str = Tenant):
        """Replay what was served, including every item's propensity."""
        row = svc.get_decision(tenant, decision_id)
        if row is None:
            raise _error(404, "not_found", f"no decision {decision_id!r}")
        return row


    # -- observe -----------------------------------------------------------

    @app.post("/v1/signals")
    def signals(req: SignalsRequest, tenant: str = Tenant):
        timer = Timer()
        now = time.time()
        warnings: list[str] = []
        no_id = sum(1 for s in req.signals if not s.signal_id)
        if no_id:
            warnings.append(
                f"{no_id} signal(s) carry no signal_id, so a redelivery of them cannot "
                f"be de-duplicated; send a stable id per event to make ingestion idempotent")
        ms_like = sum(1 for s in req.signals if s.ts is not None and s.ts > 1e11)
        if ms_like:
            warnings.append(
                f"{ms_like} signal(s) have ts > 1e11, which looks like milliseconds; "
                f"ts is epoch SECONDS and a wrong unit silently distorts time-based "
                f"uncertainty growth")

        rows = [{"user_id": s.user, "item_id": s.item, "outcome": s.outcome,
                 "ts": s.ts if s.ts is not None else now,
                 "signal_id": s.signal_id, "propensity": s.propensity,
                 "decision_id": s.decision_id,
                 "policy_id": s.policy_id, "context": s.context or {}}
                for s in req.signals]
        res = svc.observe(tenant, rows, now=now)
        if res.missing_propensity:
            warnings.append(
                f"{res.missing_propensity} accepted signal(s) have neither propensity "
                f"nor a resolvable decision_id; they train the model but cannot be used "
                f"for off-policy evaluation. Echo back decision_id from /v1/next")
        log_event("signals", tenant=tenant, accepted=res.accepted,
                  duplicates=res.duplicates, unknown=res.unknown_items,
                  backfilled=res.backfilled_propensity,
                  missing_propensity=res.missing_propensity,
                  latency_ms=round(timer.ms, 2))
        return {"accepted": res.accepted, "duplicates": res.duplicates,
                "unknown_items": res.unknown_items,
                "backfilled_propensity": res.backfilled_propensity,
                "missing_propensity": res.missing_propensity,
                "warnings": warnings}

    @app.post("/v1/items")
    def items(req: ItemsRequest, tenant: str = Tenant):
        objs = [Item(id=i.id, tag_weights=i.tags or {},
                     difficulty_prior=i.difficulty_prior, attrs=i.attrs or {})
                for i in req.items]
        out = svc.register_items(tenant, objs)
        log_event("items", tenant=tenant, **out)
        return out

    @app.get("/v1/items")
    def list_items(tenant: str = Tenant,
                   after: str | None = Query(default=None,
                                             description="cursor: last item_id of the previous page"),
                   limit: int = Query(default=200, ge=1, le=1000)):
        """Read back the catalogue.

        Registration is a write; without this the caller has no way to confirm what
        landed, how tags were parsed, or whether an id collided and overwrote
        something. Keyset cursor rather than offset so pages stay stable while the
        catalogue is being written.
        """
        return svc.list_items(tenant, after=after, limit=limit)

    @app.get("/v1/items/{item_id}")
    def get_item(item_id: str, tenant: str = Tenant):
        page = svc.store.get_items(tenant, [item_id])
        if not page:
            raise HTTPException(status_code=404, detail=f"no item {item_id!r}")
        it = page[0]
        return {"id": it.id, "tags": dict(it.tag_weights),
                "difficulty_prior": it.difficulty_prior, "attrs": dict(it.attrs)}

    # -- policies (L3) -----------------------------------------------------

    @app.post("/v1/policies")
    def put_policy(doc: PolicyIn, tenant: str = Tenant):
        """Register an L3 policy.

        Validated here, at registration, with the same rules the shipped catalogue
        passes at load -- so a policy that would void the chooser's guarantee is
        refused while someone is looking at it, not on the first request that
        happens to name it.
        """
        out = svc.register_policy(tenant, doc.model_dump(exclude_none=True),
                                 now=time.time())
        log_event("policy_registered", tenant=tenant, policy_ref=out["policy_ref"])
        return out

    @app.get("/v1/policies")
    def list_policies(tenant: str = Tenant):
        return {"policies": svc.list_policies(tenant)}

    @app.get("/v1/policies/{policy_ref}")
    def get_policy(policy_ref: str, tenant: str = Tenant):
        row = svc.get_policy(tenant, policy_ref)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no policy {policy_ref!r}")
        return row

    @app.delete("/v1/policies/{policy_ref}")
    def delete_policy(policy_ref: str, tenant: str = Tenant):
        if not svc.delete_policy(tenant, policy_ref):
            raise HTTPException(status_code=404, detail=f"no policy {policy_ref!r}")
        return {"deleted": policy_ref}


    # -- profile & subject rights -----------------------------------------

    @app.get("/v1/profile/{user}")
    def profile(user: str, tenant: str = Tenant,
                locale: str | None = Query(default=None),
                accept_language: str | None = Header(default=None)):
        out = svc.profile(tenant, user, now=time.time())
        hint = render.hint(out.get("fallback_reason"), _locale(locale, accept_language))
        return {**out, "hint": hint}


    @app.get("/v1/profile/{user}/export")
    def export_profile(user: str, tenant: str = Tenant):
        """Everything held about one user, for a data-access request."""
        out = store.export_user(tenant, user)
        log_event("export_user", tenant=tenant, user=user,
                  signals=len(out["signals"]), decisions=len(out["decisions"]))
        return out

    @app.delete("/v1/profile/{user}")
    def delete_profile(user: str, tenant: str = Tenant):
        """Erase a user's record.

        Item-side parameters (difficulty, slope) are population aggregates and are
        not reverted -- they hold no personal data and cannot be attributed to an
        individual. Said in the response rather than buried in a changelog, because
        the distinction between erasing a record and unlearning a contribution is
        exactly what a compliance reviewer will ask about.
        """
        deleted = store.delete_user(tenant, user)
        log_event("delete_user", tenant=tenant, user=user, **deleted)
        return {"deleted": deleted,
                "note": "item-level aggregates (difficulty, discrimination) are "
                        "population statistics and are not reverted"}

    # -- operations --------------------------------------------------------

    @app.post("/v1/admin/purge")
    def purge(req: PurgeRequest | None = None, tenant: str = Tenant,
              authorization: str | None = Header(default=None)):
        """Apply the retention policy. Body optional; defaults are the TTLs.

        Gated on the operator token when one is configured. A tenant key alone
        should not be able to delete that tenant's whole audit trail and
        calibration history in one unconfirmed call -- the keys handed out for
        reading and reporting are the same keys.
        """
        req = req or PurgeRequest()
        if admin_token and not _bearer_matches(authorization, admin_token):
            raise _error(403, "forbidden",
                         "purge requires the operator token in Authorization: Bearer")
        out = store.purge(tenant, now=time.time(),
                          prediction_ttl_days=req.prediction_ttl_days,
                          signal_ttl_days=req.signal_ttl_days,
                          decision_ttl_days=req.decision_ttl_days,
                          dry_run=req.dry_run)
        log_event("purge", tenant=tenant, dry_run=req.dry_run, **out)
        return {"deleted": out, "dry_run": req.dry_run}


    @app.get("/v1/goals")
    def goals(tenant: str = Tenant):
        """Discoverability for the L1/L2 surface: what goals exist, which
        adjectives describe each one (derived from the utility itself), and the
        exact accepted values for every tune knob -- so a caller can look the
        vocabulary up instead of guessing it and getting a 400."""
        tune_choices = {
            k: sorted(v for v in table if not str(v).startswith("_"))
            for k, table in catalog.tune_maps.items()
        }
        return {"default": catalog.default_goal,
                "calibrated": catalog.calibrated,
                "goals": {g: catalog.describe(g) for g in catalog.goals},
                "tune": tune_choices,
                "locales": render.available(),
                "policies": [p["policy_ref"] for p in svc.list_policies(tenant)]}


    @app.get("/healthz")
    async def healthz():
        """Liveness. Must NOT touch the database: if it did, a slow query would
        get the process killed instead of merely marked unready.

        ``async`` for the same reason, and it is what makes
        ``ADAPTIVE_MAX_CONCURRENCY`` safe to set: a sync endpoint runs in the
        threadpool, so a narrow cap would park the liveness probe behind however
        many decides are in flight and the orchestrator would restart a process
        that was merely busy. Both values it returns are plain attributes read
        once at startup, so there is nothing here to await."""
        return {"ok": True, "model_version": svc.model_version,
                "schema_version": store.schema_version}

    @app.get("/readyz")
    def readyz():
        """Readiness. Actually consults the dependency, because a probe that
        answers without checking is reporting health it has not verified."""
        try:
            store.ping()
        except Exception as exc:                     # degraded, not crashed
            metrics.incr("readyz_fail")
            return JSONResponse(status_code=503,
                                content={"ready": False, "error": type(exc).__name__})
        return {"ready": True, "schema_version": store.schema_version}

    def _scrape_scope(request: Request, authorization: str | None,
                      x_api_key: str | None) -> str | None:
        """Who is scraping, and therefore what they may see.

        Returns ``None`` when unauthorised, ``""`` for an operator holding the
        scrape token (the whole process), or the tenant id for an API key. A key
        must not see the whole snapshot: counters are labelled by tenant, so an
        unfiltered scrape hands one tenant every other tenant's traffic volume,
        degradation rate and throttling.
        """
        if scrape_token:
            return "" if _bearer_matches(authorization, scrape_token) else None
        return request.app.state.keys.resolve(x_api_key)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_openmetrics(request: Request,
                            authorization: str | None = Header(default=None),
                            x_api_key: str | None = Header(default=None)):
        """OpenMetrics text, so Prometheus can scrape it without a translator.

        Two caveats stated rather than implied: the numbers are per process, so N
        replicas give N partial views and the online-calibration window is split
        across them; and scraping is authenticated (bearer token if configured,
        otherwise an API key) because per-tenant counters are tenant information.
        An API key sees only its own tenant's series; whole-process scraping needs
        the operator token.
        """
        scope = _scrape_scope(request, authorization, x_api_key)
        if scope is None:
            raise _error(401, "unauthorized", "metrics scrape not authorised")
        return render_openmetrics(metrics.snapshot(tenant=scope or None))

    @app.get("/metrics.json")
    def metrics_json(request: Request,
                     authorization: str | None = Header(default=None),
                     x_api_key: str | None = Header(default=None)):
        """Same data with the reliability table intact -- the per-bin breakdown is
        what tells you *how* calibration is drifting, and it does not fit the flat
        OpenMetrics shape. Scoped exactly like ``/metrics``."""
        scope = _scrape_scope(request, authorization, x_api_key)
        if scope is None:
            raise _error(401, "unauthorized", "metrics scrape not authorised")
        return metrics.snapshot(tenant=scope or None)


    return app


_ORDER = {"high": 2, "medium": 1, "low": 0}


def _min_conf(a: str, b: str) -> str:
    return a if _ORDER[a] <= _ORDER[b] else b
