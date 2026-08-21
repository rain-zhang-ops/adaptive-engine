"""HTTP contract tests.

The API layer's job is authentication, translation and rendering, and each of
those has a failure mode that only shows up over HTTP:

* a missing or wrong key must be 401 and must never fall back to a default tenant
* exceeding the rate limit must be 429, not a slow success
* a bad goal is the caller's mistake -- 400, not 500
* degradation (cold user, empty catalogue, impossible constraints) must be 200
  with ``confidence`` and ``fallback_reason``, because a 5xx here is the caller's
  outage
* ``propensity`` must be echoed on the wire, since it cannot be reconstructed
  after the fact
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api import ApiKeyRegistry, RateLimiter, create_app, render_why
from engine.config import ConfigError, load_runtime
from engine.simulator import SimConfig, Simulator
from engine.store import SqliteStore

KEY_A, KEY_B = "key-alpha", "key-beta"


@pytest.fixture(scope="module")
def sim():
    return Simulator(SimConfig(n_items=120, n_users=30))


@pytest.fixture
def client(tmp_path):
    keys = ApiKeyRegistry()
    keys.add(KEY_A, "tenantA")
    keys.add(KEY_B, "tenantB")
    store = SqliteStore(tmp_path / "api.db")
    app = create_app(store=store, keys=keys, limiter=RateLimiter(rate_per_sec=1e6, burst=1e6))
    return TestClient(app)


def _hdr(key=KEY_A):
    return {"X-API-Key": key}


def _load(client, sim, key=KEY_A, n=80):
    items = [{"id": it.id, "tags": dict(it.tag_weights),
              "difficulty_prior": it.difficulty_prior, "attrs": dict(it.attrs)}
             for it in sim.items[:n]]
    r = client.post("/v1/items", json={"items": items}, headers=_hdr(key))
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# auth & limits
# ---------------------------------------------------------------------------


def test_missing_key_is_401(client):
    assert client.post("/v1/next", json={"user": "u1", "count": 3}).status_code == 401


def test_wrong_key_is_401_and_does_not_fall_back(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"user": "u1", "count": 3},
                    headers={"X-API-Key": "not-a-key"})
    assert r.status_code == 401


def test_rate_limit_returns_429(tmp_path, sim):
    keys = ApiKeyRegistry()
    keys.add(KEY_A, "tenantA")
    app = create_app(store=SqliteStore(tmp_path / "rl.db"), keys=keys,
                     limiter=RateLimiter(rate_per_sec=0.0, burst=2))
    c = TestClient(app)
    codes = [c.get("/v1/goals", headers=_hdr()).status_code for _ in range(4)]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


def test_healthz_needs_no_key(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_config_file_beats_the_environment(tmp_path, monkeypatch):
    """The file is declared intent; an environment variable is whatever a shell
    happened to hold. If the environment won, nothing in a checked-in config could
    be trusted -- a stray ADAPTIVE_* in an operator's profile would silently
    override it."""
    path = tmp_path / "adaptive.yaml"
    path.write_text("recall_limit: 300\nrate_per_sec: 7.5\ndb: from-file.db\n",
                    encoding="utf-8")
    monkeypatch.setenv("ADAPTIVE_CONFIG", str(path))
    monkeypatch.setenv("ADAPTIVE_RATE_PER_SEC", "999")
    monkeypatch.setenv("ADAPTIVE_DB", "from-env.db")
    monkeypatch.setenv("WORKERS", "4")            # not set in the file

    rc = load_runtime()
    assert (rc.recall_limit, rc.rate_per_sec, rc.db) == (300, 7.5, "from-file.db")
    assert rc.workers == 4                        # env still fills what the file omits
    assert rc.burst == 200.0                      # and defaults fill the rest


def test_config_rejects_what_it_cannot_honour(tmp_path, monkeypatch):
    path = tmp_path / "adaptive.yaml"
    monkeypatch.setenv("ADAPTIVE_CONFIG", str(path))

    path.write_text("recall_limmit: 300\n", encoding="utf-8")     # typo
    with pytest.raises(ConfigError):
        load_runtime()

    path.write_text("recall_limit: 0\n", encoding="utf-8")        # scores nothing
    with pytest.raises(ConfigError):
        load_runtime()

    path.write_text("max_concurrency: fast\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_runtime()

    monkeypatch.setenv("ADAPTIVE_CONFIG", str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError):             # named by hand, so absence is a bug
        load_runtime()


def test_in_memory_database_forces_a_single_worker(tmp_path, monkeypatch):
    """Each worker would open its own private empty database, so one tenant's data
    would be split across processes."""
    path = tmp_path / "adaptive.yaml"
    path.write_text("db: ':memory:'\nworkers: 8\n", encoding="utf-8")
    monkeypatch.setenv("ADAPTIVE_CONFIG", str(path))

    rc = load_runtime()
    assert rc.workers == 1
    assert any("forced to 1" in n for n in rc.notes)      # not silent


def test_recall_limit_from_config_reaches_the_engine(tmp_path, monkeypatch):
    path = tmp_path / "adaptive.yaml"
    path.write_text("recall_limit: 111\nrecall_tags: 3\n", encoding="utf-8")
    monkeypatch.setenv("ADAPTIVE_CONFIG", str(path))
    keys = ApiKeyRegistry()
    keys.add(KEY_A, "tenantA")
    app = create_app(store=SqliteStore(tmp_path / "rc.db"), keys=keys,
                     limiter=RateLimiter(rate_per_sec=1e6, burst=1e6))
    assert app.state.service.cfg.recall_limit == 111
    assert app.state.service.cfg.recall_tags == 3


def test_concurrency_cap_is_validated_before_serving(tmp_path, monkeypatch):
    keys = ApiKeyRegistry()
    keys.add(KEY_A, "tenantA")

    def app():
        return create_app(store=SqliteStore(tmp_path / "cc.db"), keys=keys,
                          limiter=RateLimiter(rate_per_sec=1e6, burst=1e6))

    monkeypatch.setenv("ADAPTIVE_MAX_CONCURRENCY", "4")
    with TestClient(app()) as c:                     # `with` runs lifespan
        assert c.get("/healthz").status_code == 200

    # A typo would otherwise degrade capacity silently for the process's life.
    monkeypatch.setenv("ADAPTIVE_MAX_CONCURRENCY", "lots")
    with pytest.raises(ConfigError):
        app()




# ---------------------------------------------------------------------------
# the four levels of disclosure
# ---------------------------------------------------------------------------


def test_L0_zero_config_call_works(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"user": "u1", "count": 5}, headers=_hdr())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["goal"]                       # fell back to the default goal
    assert len(body["results"][0]["items"]) == 5
    assert body["meta"]["model_version"]
    assert body["meta"]["policy_id"]


def test_L2_tune_is_reflected_in_meta(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"user": "u1", "count": 5, "goal": "practice_weak",
                                      "tune": {"difficulty": "brutal", "focus": "narrow"}},
                    headers=_hdr())
    assert r.status_code == 200, r.text
    applied = r.json()["meta"]["applied_tune"]
    assert "difficulty" in applied and "focus" in applied


def test_unknown_goal_is_400_not_500(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"user": "u1", "count": 3, "goal": "does_not_exist"},
                    headers=_hdr())
    assert r.status_code == 400
    assert r.json()["error"] == "policy"


def test_user_and_users_are_mutually_exclusive(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"count": 3}, headers=_hdr())
    assert r.status_code == 422
    r2 = client.post("/v1/next", json={"user": "a", "users": ["b"], "count": 3},
                     headers=_hdr())
    assert r2.status_code == 422


def test_batch_users(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"users": ["u1", "u2", "u3"], "count": 4},
                    headers=_hdr())
    assert r.status_code == 200
    assert [x["user"] for x in r.json()["results"]] == ["u1", "u2", "u3"]


# ---------------------------------------------------------------------------
# soft failure
# ---------------------------------------------------------------------------


def test_empty_catalogue_degrades_with_200(client):
    r = client.post("/v1/next", json={"user": "u1", "count": 5}, headers=_hdr())
    assert r.status_code == 200
    assert r.json()["confidence"] == "low"
    assert r.json()["fallback_reason"] == "empty_candidate_pool"


def test_impossible_quota_degrades_with_200(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={
        "user": "u1", "count": 5,
        "quota": [{"group_by": "attrs.kind", "counts": {"nonexistent_kind": 3}}],
    }, headers=_hdr())
    assert r.status_code == 200
    assert r.json()["confidence"] == "low"
    assert r.json()["fallback_reason"] == "constraints_unsatisfiable"


def test_short_catalogue_is_insufficient_not_empty(client, sim):
    """A catalogue with fewer items than `count` must not report an EMPTY pool.
    The two send whoever is debugging in completely different directions."""
    _load(client, sim, n=3)
    body = client.post("/v1/next", json={"user": "u1", "count": 10},
                       headers=_hdr()).json()
    assert body["confidence"] == "low"
    assert body["fallback_reason"] == "insufficient_candidates"
    assert len(body["results"][0]["items"]) == 3     # still returns what it has


def test_cold_user_is_flagged(client, sim):

    _load(client, sim)
    body = client.post("/v1/next", json={"user": "brand_new", "count": 5},
                       headers=_hdr()).json()
    assert body["confidence"] in ("medium", "low")
    assert body["fallback_reason"] == "cold_start_no_signals"



# ---------------------------------------------------------------------------
# round trip: serve -> report -> profile
# ---------------------------------------------------------------------------


def test_propensity_is_echoed_and_accepted_back(client, sim):
    _load(client, sim)
    served = client.post("/v1/next", json={"user": "u1", "count": 6, "goal": "explore"},
                         headers=_hdr()).json()
    items = served["results"][0]["items"]
    assert all("propensity" in i and i["propensity"] > 0 for i in items)

    signals = [{"user": "u1", "item": i["id"], "outcome": 1.0 if n % 2 else 0.0,
                "ts": 100.0 + n, "signal_id": f"sig-{n}",
                "propensity": i["propensity"], "policy_id": served["meta"]["policy_id"]}
               for n, i in enumerate(items)]
    r = client.post("/v1/signals", json={"signals": signals}, headers=_hdr())
    assert r.status_code == 200
    assert r.json()["accepted"] == len(items)

    # replay -> all duplicates, state untouched
    r2 = client.post("/v1/signals", json={"signals": signals}, headers=_hdr())
    body = r2.json()
    assert (body["accepted"], body["duplicates"], body["unknown_items"]) == (
        0, len(items), 0)


    prof = client.get("/v1/profile/u1", headers=_hdr()).json()
    assert prof["known"] is True
    assert prof["weakest"] and prof["strongest"]


def test_unknown_item_in_signals_is_counted_not_fatal(client, sim):
    _load(client, sim)
    r = client.post("/v1/signals", json={"signals": [
        {"user": "u1", "item": "no-such-item", "outcome": 1.0, "ts": 1.0}]},
        headers=_hdr())
    assert r.status_code == 200
    assert r.json()["unknown_items"] == 1


def test_online_calibration_appears_in_metrics(client, sim):
    _load(client, sim)
    served = client.post("/v1/next", json={"user": "u9", "count": 6},
                         headers=_hdr()).json()
    items = served["results"][0]["items"]
    client.post("/v1/signals", json={"signals": [
        {"user": "u9", "item": i["id"], "outcome": 1.0, "ts": 200.0 + n,
         "signal_id": f"c-{n}", "propensity": i["propensity"]}
        for n, i in enumerate(items)]}, headers=_hdr())

    snap = client.get("/metrics.json", headers=_hdr()).json()
    assert snap["calibration"]["n"] == len(items)
    assert snap["calibration"]["ece"] is not None
    assert "decide" in " ".join(snap["counters"])


def test_openmetrics_scrape_is_text_and_authenticated(client, sim):
    _load(client, sim)
    client.post("/v1/next", json={"user": "u9", "count": 3}, headers=_hdr())

    # no credential at all -> refused; counters are tenant information
    assert client.get("/metrics").status_code == 401

    r = client.get("/metrics", headers=_hdr())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "# TYPE engine_decide_total counter" in body
    assert "engine_latency_ms{op=\"decide\",quantile=\"0.5\"}" in body
    assert body.rstrip().endswith("# EOF")


def test_openmetrics_bearer_token_replaces_api_key(tmp_path, sim):
    keys = ApiKeyRegistry()
    keys.add(KEY_A, "tenantA")
    app = create_app(store=SqliteStore(tmp_path / "m.db"), keys=keys,
                     limiter=RateLimiter(rate_per_sec=1e6, burst=1e6),
                     metrics_token="scrape-secret")
    c = TestClient(app)
    # once a scrape token is configured, an API key is no longer sufficient
    assert c.get("/metrics", headers=_hdr()).status_code == 401
    assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/metrics", headers={"Authorization": "Bearer scrape-secret"}).status_code == 200



# ---------------------------------------------------------------------------
# tenancy over HTTP
# ---------------------------------------------------------------------------


def test_tenant_isolation_over_http(client, sim):
    _load(client, sim, key=KEY_A, n=60)
    signals = [{"user": "u1", "item": sim.items[j].id, "outcome": 1.0,
                "ts": float(j), "signal_id": f"a-{j}"} for j in range(10)]
    client.post("/v1/signals", json={"signals": signals}, headers=_hdr(KEY_A))

    # tenant B: same user id, own (empty) catalogue and own (absent) state
    assert client.get("/v1/profile/u1", headers=_hdr(KEY_A)).json()["known"] is True
    assert client.get("/v1/profile/u1", headers=_hdr(KEY_B)).json()["known"] is False
    b = client.post("/v1/next", json={"user": "u1", "count": 3},
                    headers=_hdr(KEY_B)).json()
    assert b["fallback_reason"] == "empty_candidate_pool"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_why_is_a_sentence_not_a_score_dump():
    s = render_why({"fit": 0.8, "structure": -0.1, "explore": 0.0}, 0.68)
    assert "68" in s and "{" not in s
    assert render_why({"explore": 1.0}, 0.5) != s        # different reason, different text


def test_goals_endpoint_is_self_describing(client):
    body = client.get("/v1/goals", headers=_hdr()).json()
    assert body["default"] in body["goals"]
    assert body["calibrated"] is False
    for name, d in body["goals"].items():
        assert d["label"] and d["intent"]
    # the exact accepted values, so a caller can look the vocabulary up instead
    # of guessing it and getting a 400
    assert "moderate" in body["tune"]["difficulty"]
    assert "zh" in body["locales"]


def test_locale_selects_language_and_unknown_locale_falls_back(client, sim):
    _load(client, sim)
    en = client.post("/v1/next", json={"user": "u1", "count": 3, "locale": "en"},
                     headers=_hdr()).json()
    zh = client.post("/v1/next", json={"user": "u1", "count": 3, "locale": "zh-CN"},
                     headers=_hdr()).json()
    de = client.post("/v1/next", json={"user": "u1", "count": 3, "locale": "de"},
                     headers=_hdr()).json()

    assert "predicted success" in en["results"][0]["items"][0]["why"]
    assert "预计成功率" in zh["results"][0]["items"][0]["why"]
    # an unsupported locale is served in the default one, not refused
    assert de["results"][0]["items"][0]["why"] == zh["results"][0]["items"][0]["why"]
    # the hint follows the same locale as the why
    assert "no history yet" in en["hint"]


# ---------------------------------------------------------------------------
# closing the loop by decision id
# ---------------------------------------------------------------------------


def test_decision_is_replayable_by_id(client, sim):
    """The caller should not have to keep a copy of what was served."""
    _load(client, sim)
    served = client.post("/v1/next", json={"user": "u1", "count": 4},
                         headers=_hdr()).json()
    did = served["results"][0]["decision_id"]
    assert did

    row = client.get(f"/v1/decisions/{did}", headers=_hdr()).json()
    assert row["user_id"] == "u1"
    assert row["policy_id"] == served["meta"]["policy_id"]
    replayed = {c["item_id"]: c["propensity"] for c in row["payload"]["chosen"]}
    assert replayed == {i["id"]: i["propensity"] for i in served["results"][0]["items"]}

    assert client.get("/v1/decisions/nope", headers=_hdr()).status_code == 404
    # another tenant's decision id is simply not there
    assert client.get(f"/v1/decisions/{did}", headers=_hdr(KEY_B)).status_code == 404


def test_propensity_is_backfilled_from_decision_id(client, sim):
    """Reporting only (decision_id, item, outcome) must still be IPS-usable."""
    _load(client, sim)
    served = client.post("/v1/next", json={"user": "u2", "count": 5, "goal": "explore"},
                         headers=_hdr()).json()
    did = served["results"][0]["decision_id"]
    items = served["results"][0]["items"]

    r = client.post("/v1/signals", json={"signals": [
        {"user": "u2", "item": i["id"], "outcome": 1.0, "ts": 300.0 + n,
         "signal_id": f"{did}:{i['id']}", "decision_id": did}
        for n, i in enumerate(items)]}, headers=_hdr())
    body = r.json()
    assert body["accepted"] == len(items)
    assert body["backfilled_propensity"] == len(items)
    assert body["missing_propensity"] == 0

    stored = client.get("/v1/profile/u2/export", headers=_hdr()).json()["signals"]
    assert all(s["propensity"] is not None for s in stored)


def test_signal_without_propensity_or_decision_is_warned_not_dropped(client, sim):
    _load(client, sim)
    r = client.post("/v1/signals", json={"signals": [
        {"user": "u3", "item": sim.items[0].id, "outcome": 1.0, "ts": 10.0}]},
        headers=_hdr())
    body = r.json()
    assert body["accepted"] == 1                 # it still trains the model
    assert body["missing_propensity"] == 1
    joined = " ".join(body["warnings"])
    assert "signal_id" in joined                 # no idempotency key either
    assert "decision_id" in joined               # and how to fix the propensity gap


def test_ts_is_optional_and_millisecond_ts_is_warned(client, sim):
    _load(client, sim)
    ok = client.post("/v1/signals", json={"signals": [
        {"user": "u4", "item": sim.items[1].id, "outcome": 1.0, "signal_id": "no-ts"}]},
        headers=_hdr())
    assert ok.status_code == 200 and ok.json()["accepted"] == 1

    ms = client.post("/v1/signals", json={"signals": [
        {"user": "u4", "item": sim.items[2].id, "outcome": 1.0,
         "ts": 1.7e12, "signal_id": "ms-ts"}]}, headers=_hdr())
    assert ms.status_code == 200
    assert any("milliseconds" in w for w in ms.json()["warnings"])


# ---------------------------------------------------------------------------
# L3: tenant-registered policies
# ---------------------------------------------------------------------------


_POLICY = {
    "id": "night-drill",
    "label": "night drill",
    "extends": "practice_weak",
    "utility": {"rho": {"kind": "peak", "target": 0.62}, "explore_floor": 0.10},
    "constraints": {"max_per_tag": 2},
}


def test_policy_ref_is_registered_validated_and_used(client, sim):
    _load(client, sim)
    reg = client.post("/v1/policies", json=_POLICY, headers=_hdr())
    assert reg.status_code == 200, reg.text
    assert reg.json()["policy_ref"] == "night-drill"
    # no provenance.status -> the same bar the shipped catalogue is held to
    assert any("provenance" in w for w in reg.json()["warnings"])

    served = client.post("/v1/next", json={"user": "u1", "count": 6,
                                           "policy_ref": "night-drill"},
                         headers=_hdr())
    assert served.status_code == 200, served.text
    meta = served.json()["meta"]
    # the ref itself is the audit identity -- more useful than a hash of a
    # document stored elsewhere
    assert meta["policy_id"] == "night-drill"
    assert meta["policy_ref"] == "night-drill"

    assert [p["policy_ref"] for p in client.get("/v1/policies",
                                                headers=_hdr()).json()["policies"]] \
        == ["night-drill"]
    assert client.get("/v1/policies/night-drill",
                      headers=_hdr()).json()["doc"]["extends"] == "practice_weak"
    assert "night-drill" in client.get("/v1/goals", headers=_hdr()).json()["policies"]

    # another tenant cannot see or use it
    assert client.get("/v1/policies/night-drill", headers=_hdr(KEY_B)).status_code == 404
    assert client.delete("/v1/policies/night-drill", headers=_hdr()).status_code == 200
    assert client.get("/v1/policies/night-drill", headers=_hdr()).status_code == 404


def test_policy_that_would_void_the_guarantee_is_refused_at_registration(client):
    """A negative structure weight destroys submodularity. Refuse it while
    someone is looking at it, not on the first request that names it."""
    r = client.post("/v1/policies", json={
        "id": "bad-weight",
        "utility": {"rho": {"kind": "peak", "target": 0.7},
                    "structure": {"kind": "diversify", "weight": -0.5}},
    }, headers=_hdr())
    assert r.status_code == 400
    assert r.json()["error"] == "policy"
    assert "structure_weight_nonnegative" in r.json()["detail"]
    # and it was not stored
    assert client.get("/v1/policies/bad-weight", headers=_hdr()).status_code == 404


def test_declared_but_unimplemented_policy_fields_are_refused(client):
    r = client.post("/v1/policies", json={
        "id": "other-believe", "extends": "practice_weak", "believe": "bandit",
    }, headers=_hdr())
    assert r.status_code == 400 and "mtor" in r.json()["detail"]

    r2 = client.post("/v1/policies", json={
        "id": "seen-filter", "extends": "practice_weak",
        "constraints": {"exclude_seen_within_days": 7},
    }, headers=_hdr())
    assert r2.status_code == 400 and "not implemented" in r2.json()["detail"]


def test_unknown_policy_ref_is_400_and_lists_what_exists(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"user": "u1", "count": 3,
                                      "policy_ref": "never-registered"},
                    headers=_hdr())
    assert r.status_code == 400
    assert "never-registered" in r.json()["detail"]


def test_goal_and_policy_ref_together_is_422(client, sim):
    _load(client, sim)
    r = client.post("/v1/next", json={"user": "u1", "count": 3, "goal": "explore",
                                      "policy_ref": "night-drill"}, headers=_hdr())
    assert r.status_code == 422
    assert r.json()["error"] == "validation"
    assert "not both" in r.json()["detail"]


# ---------------------------------------------------------------------------
# readable rejections
# ---------------------------------------------------------------------------


def test_mistyped_tune_value_is_rejected_readably(client, sim):
    """A typo gets the near miss; a synonym gets the vocabulary.

    Edit distance catches 'moderat' and cannot catch 'medium' -- they are 0.29
    similar. Lowering the cutoff far enough to match a synonym would start
    suggesting unrelated words, so the choices list is always printed and the
    "did you mean" is an extra for true typos rather than the mechanism.
    """
    _load(client, sim)

    typo = client.post("/v1/next", json={"user": "u1", "count": 3,
                                         "tune": {"difficulty": "moderat"}},
                       headers=_hdr())
    assert typo.status_code == 400
    assert "did you mean 'moderate'" in typo.json()["detail"]

    synonym = client.post("/v1/next", json={"user": "u1", "count": 3,
                                            "tune": {"difficulty": "medium"}},
                          headers=_hdr())
    assert synonym.status_code == 400
    assert "moderate" in synonym.json()["detail"]        # the full choice list

    bad_key = client.post("/v1/next", json={"user": "u1", "count": 3,
                                            "tune": {"freshnes": 0.2}},
                          headers=_hdr())
    # a mistyped knob must not be silently dropped -- pydantic's default would
    # have returned 200 and no freshness
    assert bad_key.status_code == 400
    assert "did you mean 'freshness'" in bad_key.json()["detail"]


def test_schema_rejection_uses_the_same_envelope_as_a_policy_rejection(client):
    r = client.post("/v1/next", json={"user": "u1", "count": 0}, headers=_hdr())
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation"
    assert body["problems"] and "count" in body["detail"]


# ---------------------------------------------------------------------------
# catalogue read-back
# ---------------------------------------------------------------------------


def test_registration_counts_created_and_updated(client, sim):
    first = _load(client, sim, n=10)
    assert (first["created"], first["updated"], first["total"]) == (10, 0, 10)
    again = _load(client, sim, n=10)
    assert (again["created"], again["updated"]) == (0, 10)
    assert first["tags"] > 0


def test_catalogue_is_readable_back_with_a_cursor(client, sim):
    _load(client, sim, n=7)
    page = client.get("/v1/items?limit=3", headers=_hdr()).json()
    assert page["count"] == 3 and page["total"] == 7 and page["next_after"]

    seen = list(page["items"])
    after = page["next_after"]
    while after:
        nxt = client.get(f"/v1/items?limit=3&after={after}", headers=_hdr()).json()
        seen.extend(nxt["items"])
        after = nxt["next_after"]
    assert len(seen) == 7
    assert len({i["id"] for i in seen}) == 7          # no repeats across pages

    one = client.get(f"/v1/items/{seen[0]['id']}", headers=_hdr()).json()
    assert one["tags"] == seen[0]["tags"]
    assert client.get("/v1/items/no-such-item", headers=_hdr()).status_code == 404


# ---------------------------------------------------------------------------
# actionable degradation
# ---------------------------------------------------------------------------


def test_degraded_results_say_what_to_do_about_it(client, sim):
    """A fallback_reason without a next step leaves the caller guessing."""
    empty = client.post("/v1/next", json={"user": "u1", "count": 3},
                        headers=_hdr()).json()
    assert empty["fallback_reason"] == "empty_candidate_pool"
    assert "/v1/items" in empty["hint"]

    _load(client, sim)
    cold = client.post("/v1/next", json={"user": "never-seen", "count": 3},
                       headers=_hdr()).json()
    assert cold["fallback_reason"] == "cold_start_no_signals"
    assert cold["hint"]

    # a healthy result has nothing to advise
    warm = client.post("/v1/next", json={"users": ["u1"], "count": 3},
                       headers=_hdr()).json()
    if warm["confidence"] == "high":
        assert warm["hint"] is None

    prof = client.get("/v1/profile/never-seen", headers=_hdr()).json()
    assert prof["known"] is False and prof["hint"]


# ---------------------------------------------------------------------------
# the published contract
# ---------------------------------------------------------------------------


def test_openapi_yaml_describes_exactly_the_routes_that_exist(client):
    """openapi.yaml is the document a customer integrates against.

    Nothing checked it against the app before, and it had drifted: it documented
    ``PUT /v1/items`` (the server takes POST) and a 202 with ``new_tags`` (the
    server returns 200 with created/updated). A hand-maintained contract drifts
    silently, so the comparison is a test rather than a review habit.
    """
    import yaml

    with open(Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml",
              encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    documented = {
        (m.upper(), path)
        for path, ops in spec["paths"].items()
        for m in ops
        if m.lower() in ("get", "post", "put", "patch", "delete")
    }
    implemented = {
        (m, r.path)
        for r in client.app.routes
        for m in getattr(r, "methods", ()) or ()
        if m in ("GET", "POST", "PUT", "PATCH", "DELETE")
        and not r.path.startswith(("/openapi", "/docs", "/redoc"))
    }
    # FastAPI adds HEAD alongside GET; path params are named identically in both
    assert implemented - documented == set(), \
        f"undocumented routes: {sorted(implemented - documented)}"
    assert documented - implemented == set(), \
        f"documented but absent: {sorted(documented - implemented)}"
