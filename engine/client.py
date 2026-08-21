"""Python client -- the integration path, with the bookkeeping removed.

Standard library only, one file, no dependency on the engine package: copy it
into a client project and it works. That constraint is the point. An SDK that
drags numpy and FastAPI along is not something a caller will actually vendor, and
"just call the HTTP API yourself" is how every integrator re-derives the same two
mistakes:

*Losing the propensity.* It is the only number in a response that cannot be
reconstructed afterwards, and without it off-policy evaluation is silently biased
rather than broken. Here you never handle it: :meth:`Slate.report` sends the
``decision_id`` and the server looks it up.

*Forgetting idempotency.* A retried report double-counts unless it carries a
stable id. This client derives one from ``(decision_id, item)`` by default, which
is exactly the natural key -- one outcome per served item per decision.

Server-side warnings (missing propensity, millisecond timestamps, absent
idempotency keys) are surfaced through :mod:`warnings` instead of being buried in
a response field nobody reads.

    from engine.client import AdaptiveClient

    api = AdaptiveClient("http://localhost:8080", "dev-key")
    api.register_items([{"id": "i1", "tags": {"algebra": 1.0}}])

    slate = api.next("u1", count=10, goal="practice_weak")
    for item in slate.items:
        print(item["id"], item["why"])

    slate.report("i1", outcome=1.0)          # propensity handled for you
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["AdaptiveClient", "AdaptiveError", "Slate"]


class AdaptiveError(RuntimeError):
    """A 4xx/5xx from the service.

    Carries the server's own ``detail`` line, which is written to be read by a
    person -- including the "did you mean ...?" suggestion on a mistyped goal or
    tune value.
    """

    def __init__(self, status: int, detail: str, body: Any = None) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail
        self.body = body


@dataclass
class Slate:
    """One user's returned set, plus the handle needed to close the loop."""

    client: "AdaptiveClient"
    user: str
    decision_id: str
    items: list[dict[str, Any]]
    confidence: str
    fallback_reason: str | None = None
    hint: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return self.confidence != "high"

    @property
    def item_ids(self) -> list[str]:
        return [i["id"] for i in self.items]

    def report(self, item: str, outcome: float, ts: float | None = None,
               signal_id: str | None = None,
               context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Report one outcome for an item from this slate."""
        return self.report_many([{"item": item, "outcome": outcome, "ts": ts,
                                  "signal_id": signal_id, "context": context}])

    def report_many(self, outcomes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Report several outcomes from this slate in one request.

        Each entry needs ``item`` and ``outcome``; ``ts``, ``signal_id`` and
        ``context`` are optional. The decision id is attached automatically, which
        is what makes the propensity the server's problem rather than yours.
        """
        rows = []
        for o in outcomes:
            item = o["item"]
            rows.append({
                "user": self.user,
                "item": item,
                "outcome": float(o["outcome"]),
                "ts": o.get("ts"),
                # (decision, item) is the natural key: one outcome per served item
                # per decision. A retry therefore de-duplicates by construction.
                "signal_id": o.get("signal_id") or f"{self.decision_id}:{item}",
                "decision_id": self.decision_id,
                "context": dict(o.get("context") or {}) or None,
            })
        return self.client.report(rows)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects instead of following them.

    ``urlopen`` follows 30x by default and carries the original headers to the new
    location, which would send ``X-API-Key`` to whatever host the redirect names.
    A recommendation endpoint has no reason to redirect, so the safe reading of one
    is that something is wrong.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AdaptiveError(code, f"refusing to follow redirect to {newurl!r}; "
                                  f"credentials must not leave the configured host", {})


class AdaptiveClient:
    def __init__(self, base_url: str, api_key: str, locale: str | None = None,
                 timeout: float = 10.0, warn: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"base_url must be an absolute http(s) URL, got {base_url!r}")
        self.api_key = api_key
        self.locale = locale
        self.timeout = timeout
        self.warn = warn
        """Whether to re-raise server warnings through :mod:`warnings`. On by
        default: the warnings exist because the conditions they describe are
        invisible until months later, when the data is already unusable."""
        self._opener = urllib.request.build_opener(_NoRedirect())

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, body: Any = None,
                 params: Mapping[str, Any] | None = None) -> Any:
        url = self.base_url + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("X-API-Key", self.api_key)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read()

        except urllib.error.HTTPError as e:            # 4xx / 5xx
            raw = e.read()
            try:
                parsed = json.loads(raw) if raw else {}
            except ValueError:
                parsed = {}
            detail = parsed.get("detail") or parsed.get("error") or (
                raw.decode("utf-8", "replace")[:200] if raw else e.reason)
            raise AdaptiveError(e.code, str(detail), parsed) from None
        out = json.loads(raw) if raw else None
        self._surface_warnings(out)
        return out

    def _surface_warnings(self, payload: Any) -> None:
        if not self.warn or not isinstance(payload, Mapping):
            return
        for w in payload.get("warnings") or []:
            warnings.warn(f"adaptive-engine: {w}", stacklevel=3)
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            for w in meta.get("warnings") or []:
                warnings.warn(f"adaptive-engine policy: {w}", stacklevel=3)

    # -- catalogue ---------------------------------------------------------

    def register_items(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Upsert items. Returns created/updated counts so a re-push is auditable."""
        return self._request("POST", "/v1/items", {"items": [dict(i) for i in items]})

    def list_items(self, after: str | None = None, limit: int = 200) -> dict[str, Any]:
        return self._request("GET", "/v1/items", params={"after": after, "limit": limit})

    def get_item(self, item_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/items/{urllib.parse.quote(item_id)}")

    def iter_items(self, page: int = 200):
        """Walk the whole catalogue, following the keyset cursor."""
        after = None
        while True:
            out = self.list_items(after=after, limit=page)
            for it in out["items"]:
                yield it
            after = out.get("next_after")
            if not after:
                return

    # -- decide ------------------------------------------------------------

    def next(self, user: str, count: int = 10, goal: str | None = None,
             tune: Mapping[str, Any] | None = None,
             policy_ref: str | None = None,
             within: Mapping[str, Any] | None = None,
             exclude: Mapping[str, Any] | None = None,
             quota: Sequence[Mapping[str, Any]] | None = None,
             explain: bool = True, locale: str | None = None) -> Slate:
        out = self._next_payload([user], count, goal, tune, policy_ref, within,
                                 exclude, quota, explain, locale)
        return self._slates(out)[0]

    def next_many(self, users: Sequence[str], count: int = 10, **kw: Any) -> list[Slate]:
        """One request, one policy resolution, one catalogue read for many users."""
        out = self._next_payload(list(users), count, kw.get("goal"), kw.get("tune"),
                                 kw.get("policy_ref"), kw.get("within"),
                                 kw.get("exclude"), kw.get("quota"),
                                 kw.get("explain", True), kw.get("locale"))
        return self._slates(out)

    def _next_payload(self, users: list[str], count: int, goal, tune, policy_ref,
                      within, exclude, quota, explain, locale) -> dict[str, Any]:
        body: dict[str, Any] = {"count": count, "explain": explain}
        if len(users) == 1:
            body["user"] = users[0]
        else:
            body["users"] = users
        for k, v in (("goal", goal), ("tune", tune), ("policy_ref", policy_ref),
                     ("within", within), ("exclude", exclude), ("quota", quota),
                     ("locale", locale or self.locale)):
            if v is not None:
                body[k] = v
        return self._request("POST", "/v1/next", body)

    def _slates(self, out: Mapping[str, Any]) -> list[Slate]:
        return [
            Slate(client=self, user=r["user"], decision_id=r["decision_id"],
                  items=list(r["items"]), confidence=out.get("confidence", "low"),
                  fallback_reason=out.get("fallback_reason"),
                  hint=out.get("hint"), meta=dict(out.get("meta") or {}))
            for r in out.get("results") or []
        ]

    def decision(self, decision_id: str) -> dict[str, Any]:
        """Replay a past decision -- items, propensities, policy and model version."""
        return self._request("GET", f"/v1/decisions/{urllib.parse.quote(decision_id)}")

    # -- observe -----------------------------------------------------------

    def report(self, signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Send raw signal rows. Prefer :meth:`Slate.report`, which fills in the
        decision id and a natural idempotency key for you."""
        rows = [{k: v for k, v in dict(s).items() if v is not None} for s in signals]
        return self._request("POST", "/v1/signals", {"signals": rows})

    # -- state & discovery -------------------------------------------------

    def profile(self, user: str, locale: str | None = None) -> dict[str, Any]:
        return self._request("GET", f"/v1/profile/{urllib.parse.quote(user)}",
                             params={"locale": locale or self.locale})

    def export_user(self, user: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/profile/{urllib.parse.quote(user)}/export")

    def delete_user(self, user: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/profile/{urllib.parse.quote(user)}")

    def goals(self) -> dict[str, Any]:
        """Goals, the adjectives that describe them, and every accepted tune value."""
        return self._request("GET", "/v1/goals")

    # -- policies (L3) -----------------------------------------------------

    def put_policy(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/policies", dict(doc))

    def policies(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/policies")["policies"]

    def get_policy(self, policy_ref: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/policies/{urllib.parse.quote(policy_ref)}")

    def delete_policy(self, policy_ref: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/policies/{urllib.parse.quote(policy_ref)}")

    # -- ops ---------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/healthz")

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/readyz")
