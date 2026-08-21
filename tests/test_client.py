"""SDK tests.

The client exists to remove the two mistakes every integrator makes on their own
-- losing the propensity and forgetting idempotency -- so those two are what is
asserted here, plus the fact that server warnings actually reach the caller
instead of sitting unread in a response field.

The transport is swapped for a ``TestClient`` shim rather than mocked with canned
payloads. A mock would let the client build a wrong URL, send a wrong body shape
or read a field the server does not return, and still pass; routed through the
real app, those all fail. What is not covered here is the ``urllib`` layer
itself (headers, timeouts, HTTPError parsing) -- that needs a socket, and
``loadtest.py`` is what drives the client over one.
"""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

from engine.api import ApiKeyRegistry, RateLimiter, create_app
from engine.client import AdaptiveClient, AdaptiveError, Slate
from engine.simulator import SimConfig, Simulator
from engine.store import SqliteStore

KEY = "sdk-key"


@pytest.fixture(scope="module")
def sim():
    return Simulator(SimConfig(n_items=60, n_users=10))


@pytest.fixture
def api(tmp_path, sim):
    keys = ApiKeyRegistry()
    keys.add(KEY, "tenantSDK")
    app = create_app(store=SqliteStore(tmp_path / "sdk.db"), keys=keys,
                     limiter=RateLimiter(rate_per_sec=1e6, burst=1e6))
    http = TestClient(app)

    client = AdaptiveClient("http://testserver", KEY)
    calls: list[tuple[str, str]] = []

    def _request(method, path, body=None, params=None):
        calls.append((method, path))
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        r = http.request(method, path, json=body, params=clean,
                         headers={"X-API-Key": KEY})
        if r.status_code >= 400:
            payload = r.json()
            raise AdaptiveError(r.status_code,
                                str(payload.get("detail") or payload.get("error")),
                                payload)
        out = r.json()
        client._surface_warnings(out)
        return out

    client._request = _request           # type: ignore[method-assign]
    client.calls = calls                 # type: ignore[attr-defined]

    client.register_items([{"id": it.id, "tags": dict(it.tag_weights),
                            "difficulty_prior": it.difficulty_prior,
                            "attrs": dict(it.attrs)}
                           for it in sim.items[:40]])
    return client


def test_slate_exposes_what_a_caller_needs(api):
    slate = api.next("u1", count=5, goal="practice_weak")
    assert isinstance(slate, Slate)
    assert slate.user == "u1" and slate.decision_id
    assert len(slate.items) == 5
    assert slate.item_ids == [i["id"] for i in slate.items]
    assert slate.degraded is (slate.confidence != "high")
    assert slate.meta["policy_id"]


def test_report_closes_the_loop_without_the_caller_touching_propensity(api):
    """The propensity never appears in client code, and it is still recorded."""
    slate = api.next("u2", count=4, goal="explore")
    out = slate.report_many([{"item": i, "outcome": 1.0} for i in slate.item_ids])

    assert out["accepted"] == 4
    assert out["backfilled_propensity"] == 4          # resolved from decision_id
    assert out["missing_propensity"] == 0

    replay = api.decision(slate.decision_id)
    assert {c["item_id"] for c in replay["payload"]["chosen"]} == set(slate.item_ids)


def test_a_retried_report_is_a_no_op(api):
    """(decision, item) is the natural key, so a redelivery de-duplicates by
    construction -- the caller does not have to invent an idempotency scheme."""
    slate = api.next("u3", count=3)
    first = slate.report_many([{"item": i, "outcome": 0.0} for i in slate.item_ids])
    again = slate.report_many([{"item": i, "outcome": 0.0} for i in slate.item_ids])

    assert first["accepted"] == 3 and first["duplicates"] == 0
    assert (again["accepted"], again["duplicates"]) == (0, 3)


def test_single_report_is_the_batch_path(api):
    slate = api.next("u4", count=2)
    out = slate.report(slate.item_ids[0], outcome=1.0)
    assert out["accepted"] == 1
    assert not out["warnings"]                        # id and propensity both filled


def test_server_warnings_are_raised_not_buried(api):
    """A response field nobody reads is the same as no warning at all."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        api.report([{"user": "u5", "item": api.list_items(limit=1)["items"][0]["id"],
                     "outcome": 1.0, "ts": 1.0}])
    text = " ".join(str(w.message) for w in caught)
    assert "signal_id" in text and "decision_id" in text


def test_warnings_can_be_switched_off(api):
    api.warn = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        api.report([{"user": "u6", "item": api.list_items(limit=1)["items"][0]["id"],
                     "outcome": 1.0, "ts": 1.0}])
    assert not caught


def test_iter_items_walks_every_page(api):
    seen = [it["id"] for it in api.iter_items(page=7)]
    assert len(seen) == 40 and len(set(seen)) == 40


def test_errors_carry_the_servers_own_explanation(api):
    with pytest.raises(AdaptiveError) as e:
        api.next("u1", count=3, tune={"difficulty": "medium"})
    assert e.value.status == 400
    assert "moderate" in e.value.detail            # the suggestion survives transport


def test_l3_policy_round_trip_through_the_sdk(api):
    out = api.put_policy({"id": "sdk-policy", "extends": "practice_weak",
                          "utility": {"rho": {"kind": "peak", "target": 0.6}}})
    assert out["policy_ref"] == "sdk-policy"
    assert [p["policy_ref"] for p in api.policies()] == ["sdk-policy"]

    slate = api.next("u7", count=3, policy_ref="sdk-policy")
    assert slate.meta["policy_id"] == "sdk-policy"

    api.delete_policy("sdk-policy")
    assert api.policies() == []


def test_batch_is_one_request_for_many_users(api):
    before = len(api.calls)
    slates = api.next_many(["u8", "u9", "u10"], count=3, goal="review")
    assert [s.user for s in slates] == ["u8", "u9", "u10"]
    assert len({s.decision_id for s in slates}) == 3
    assert len(api.calls) - before == 1


def test_profile_and_erasure(api):
    slate = api.next("u11", count=3)
    slate.report(slate.item_ids[0], outcome=1.0)
    assert api.profile("u11")["known"] is True

    assert api.export_user("u11")["signals"]
    api.delete_user("u11")
    assert api.profile("u11")["known"] is False


def test_health_and_readiness_are_distinct(api):
    assert api.health()["ok"] is True
    assert api.ready()["ready"] is True
