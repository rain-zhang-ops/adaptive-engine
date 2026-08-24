"""Invariants that must hold regardless of data -- the properties a customer
would notice breaking.

These are deliberately not "does the code run" tests. Each one encodes a claim
the product makes:

* hard constraints are filters, so no score can push a forbidden item through
* the reported propensity IS the marginal inclusion probability (checked against
  Monte-Carlo, and against the analytic version used by off-policy evaluation)
* Phi stays submodular, so the (1-1/e) guarantee the chooser advertises is real
* the predicate DSL cannot execute code
* persistence is transparent: a restarted service continues, it does not reset
* tenants cannot read each other, including through a growing tag space
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from contracts.core import Constraints, Item, Quota, StructureSpec
from engine.chooser import GreedyChooser
from engine.mtor import MTOR, MTORConfig
from engine.policy import GoalCatalog, PolicyError, load_catalog
from engine.predicates import PredicateError, compile_predicate
from engine.scorer import UtilityScorer
from engine.service import EngineService
from engine.simulator import SimConfig, Simulator
from engine.store import MIGRATIONS, SqliteStore, TenantItemStore
from engine.transition import TransitionError, get_transition


@pytest.fixture(scope="module")
def sim():
    return Simulator(SimConfig(n_items=300, n_users=60))


@pytest.fixture(scope="module")
def parts(sim):
    mtor = MTOR(MTORConfig(floor_attr="floor"))
    scorer = UtilityScorer(mtor)
    cat = load_catalog()
    return mtor, scorer, cat


def _warm(sim, mtor, user=0, n=40):
    b = mtor.init(sim.users[user], sim.space)
    for j, it in enumerate(sim.sample_items(n)):
        b = mtor.update(b, sim.answer(user, it.id, j, evolve=False), it)
    return b


# ---------------------------------------------------------------------------
# constraints are filters, not penalties
# ---------------------------------------------------------------------------


def test_excluded_item_never_returned_even_when_best(sim, parts):
    """Excluding the single highest-utility item must remove it outright. If
    exclusion were a score penalty, a big enough utility would leak it back."""
    mtor, scorer, cat = parts
    u = cat.utility("practice_weak")
    b = _warm(sim, mtor)
    pool = sim.sample_items(100)
    scored = scorer.value(b, pool, u)
    best = max(scored, key=lambda s: s.utility).item_id

    dec = GreedyChooser(seed=1).solve(
        scored, pool, u, Constraints(k=5, exclude_item_ids=frozenset({best})))
    assert best not in {c.item_id for c in dec.chosen}


def test_quota_and_tag_cap_hold_together(sim, parts):
    mtor, scorer, cat = parts
    u = cat.utility("practice_weak")
    b = _warm(sim, mtor)
    pool = sim.sample_items(150)
    cons = Constraints(k=10, quotas=(Quota("attrs.kind", {"choice": 4, "blank": 2}),),
                       max_per_tag=3)
    dec = GreedyChooser(seed=2).solve(scorer.value(b, pool, u), pool, u, cons)
    by_id = {it.id: it for it in pool}
    picked = [by_id[c.item_id] for c in dec.chosen]

    kinds = Counter(it.attrs["kind"] for it in picked)
    assert kinds["choice"] >= 4 and kinds["blank"] >= 2
    tags = Counter(t for it in picked for t in it.tag_weights)
    assert max(tags.values()) <= 3


def test_infeasible_degrades_not_raises(sim, parts):
    mtor, scorer, cat = parts
    u = cat.utility("practice_weak")
    b = _warm(sim, mtor)
    pool = [it for it in sim.sample_items(80) if it.attrs["kind"] == "choice"][:15]
    dec = GreedyChooser(seed=3).solve(
        scorer.value(b, pool, u), pool, u,
        Constraints(k=5, quotas=(Quota("attrs.kind", {"essay": 3}),)))
    assert dec.confidence == "low"
    assert dec.fallback_reason == "constraints_unsatisfiable"


# ---------------------------------------------------------------------------
# propensity
# ---------------------------------------------------------------------------


def test_propensity_equals_empirical_inclusion_rate(sim, parts):
    """The reported number is used as an inverse-propensity weight, so it has to
    be the marginal inclusion probability. Verified against Monte-Carlo rather
    than against the formula that produced it."""
    mtor, scorer, cat = parts
    u = cat.utility("explore")            # explore_floor 0.30 -> real explore slots
    b = _warm(sim, mtor, user=5)
    pool = sim.sample_items(50)
    scored = scorer.value(b, pool, u)
    cons = Constraints(k=8)

    trials = 3000
    counts: Counter[str] = Counter()
    reported: dict[str, float] = {}
    for seed in range(trials):
        dec = GreedyChooser(seed=seed).solve(scored, pool, u, cons)
        for c in dec.chosen:
            if c.reasons["explore"] == 1.0:
                counts[c.item_id] += 1
                reported[c.item_id] = c.propensity

    checked = 0
    for iid, cnt in counts.items():
        if cnt < 150:
            continue
        emp = cnt / trials
        se = math.sqrt(max(emp * (1 - emp), 1e-9) / trials)
        assert abs(emp - reported[iid]) <= max(4 * se, 0.01), (iid, emp, reported[iid])
        checked += 1
    assert checked >= 3


def test_analytic_inclusion_matches_sampled(sim, parts):
    """``inclusion_probabilities`` is what off-policy evaluation consumes; it must
    agree with what ``solve`` actually does, including giving non-zero mass to
    explore-pool members that a particular draw did not pick."""
    mtor, scorer, cat = parts
    u = cat.utility("explore")
    b = _warm(sim, mtor, user=6)
    pool = sim.sample_items(50)
    scored = scorer.value(b, pool, u)
    cons = Constraints(k=8)

    analytic = GreedyChooser(seed=0).inclusion_probabilities(scored, pool, u, cons)
    trials = 3000
    counts: Counter[str] = Counter()
    for seed in range(trials):
        for c in GreedyChooser(seed=seed).solve(scored, pool, u, cons).chosen:
            counts[c.item_id] += 1

    # every item the sampler ever produced must carry positive analytic mass
    for iid in counts:
        assert analytic[iid] > 0.0, f"{iid} chosen {counts[iid]}x but analytic prob 0"
    # and the totals must agree
    for iid, p in analytic.items():
        if p <= 0.0:
            assert counts[iid] == 0
            continue
        emp = counts[iid] / trials
        se = math.sqrt(max(emp * (1 - emp), 1e-9) / trials)
        assert abs(emp - p) <= max(4 * se, 0.02), (iid, emp, p)


# ---------------------------------------------------------------------------
# submodularity
# ---------------------------------------------------------------------------


def test_diversify_marginal_is_non_increasing(sim, parts):
    """Phi must be submodular for the (1-1/e) bound to mean anything: adding a
    candidate to a larger selected set can never help more than adding it to a
    subset. Checked directly on the diversify marginal."""
    mtor, scorer, cat = parts
    b = _warm(sim, mtor, user=7)
    pool = sim.sample_items(40)
    ch = GreedyChooser(seed=0)
    spec = StructureSpec(kind="diversify", weight=1.0)

    cand = pool[0]
    growing = []
    prev = None
    for it in pool[1:12]:
        growing.append(it)
        marg = ch._structure_marginal(cand, list(growing), spec)
        if prev is not None:
            assert marg <= prev + 1e-12, f"marginal rose: {prev} -> {marg}"
        prev = marg


@pytest.mark.parametrize("spec", [
    StructureSpec(kind="diversify", weight=0.8),
    StructureSpec(kind="diversify", weight=0.8, similarity="tag_cosine"),
    StructureSpec(kind="concentrate", weight=0.8),
    StructureSpec(kind="balanced", weight=0.8, target_entropy=2.0),
])
def test_vectorised_phi_matches_scalar_reference(sim, parts, spec):
    """The hot path computes Phi's marginal for all candidates at once and keeps
    the max/sum over the selected set incrementally. That is an optimisation, so
    it owes an equality proof against the scalar definition -- silently different
    marginals would change which items are served with nothing failing."""
    mtor, scorer, cat = parts
    b = _warm(sim, mtor, user=11)
    pool = sim.sample_items(60)
    # include an untagged item: "no taxonomy" is a separate branch in both paths
    pool = list(pool) + [Item(id="untagged-1", tag_weights={}, difficulty_prior=0.0,
                              attrs={"kind": "choice", "floor": 0.25})]
    ch = GreedyChooser(seed=0)

    state = ch._structure_state(pool, spec)
    selected: list[Item] = []
    for step, chosen_item in enumerate(pool[:5]):
        fast = state.marginals()
        slow = [ch._structure_marginal(c, list(selected), spec) for c in pool]
        assert np.allclose(fast, slow, atol=1e-12), f"step {step} diverged"
        selected.append(chosen_item)
        state.add(step)


def test_negative_structure_weight_is_rejected_at_load():

    """A negative weight would flip Phi's sign and destroy submodularity, so the
    catalogue must refuse it rather than silently returning worse sets."""
    doc = {
        "goals": {"g": {"utility": {"rho": {"kind": "increasing"},
                                    "gamma": 0.0,
                                    "structure": {"kind": "diversify", "weight": -0.5},
                                    "explore_floor": 0.1}}},
        "tune_maps": {},
        "default_goal": "g",
        "validation": [{"id": k, "severity": "error"} for k in (
            "learning_goals_need_peak", "amplification_needs_explore_floor",
            "lookahead_needs_transition_model", "target_dim_weight_needs_target",
            "peak_target_in_range", "structure_weight_nonnegative")],
        # A catalogue with no provenance.status is rejected on its own terms, so
        # the fixture has to declare one to reach the structure-weight check.
        "provenance": {"status": "uncalibrated"},
    }
    with pytest.raises(PolicyError, match="structure_weight_nonnegative"):
        GoalCatalog(doc)


# ---------------------------------------------------------------------------
# policy / transition validation
# ---------------------------------------------------------------------------


def test_shipped_catalogue_loads_and_is_flagged_uncalibrated():
    cat = load_catalog()
    assert len(cat.goals) >= 5
    # Guard against someone flipping the flag without doing the work.
    assert cat.calibrated is False


def test_unknown_transition_model_rejected():
    with pytest.raises(TransitionError):
        get_transition("no_such_model")
    with pytest.raises(TransitionError):
        get_transition(None)          # gamma>0 has no default, on purpose


def test_transition_models_declare_status():
    from engine.transition import TRANSITIONS
    for name, m in TRANSITIONS.items():
        assert m.status in ("exact", "hypothesis", "supported"), name
        assert m.hypothesis, name
    # martingale must be exactly zero -- observing alone cannot raise mastery
    assert float(get_transition("info_only").mean_shift(np.array([0.5]))[0]) == 0.0


def test_unknown_tune_key_rejected():
    cat = load_catalog()
    with pytest.raises(PolicyError):
        cat.resolve("practice_weak", {"nonsense": 1})


def test_describe_is_derived_not_stored():
    """Adjectives come from reverse lookup, so they cannot contradict the utility
    they describe."""
    cat = load_catalog()
    d = cat.describe("challenge")
    assert d["focus"] == "narrow"          # concentrate structure
    assert d["difficulty"] in ("hard", "brutal", "moderate")


# ---------------------------------------------------------------------------
# predicate DSL is not eval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo x')",
    "attrs.kind == __import__('os')",
    "1 + 1 == 2",
    "attrs.kind == 'a' and attrs.x == 1",
])
def test_predicate_rejects_non_grammar(expr):
    with pytest.raises(PredicateError):
        compile_predicate(expr)


def test_missing_attribute_satisfies_not_equal():
    """Otherwise "visibility != 'embargoed'" would drop every item that simply
    has no visibility attribute, turning an optional field into a required one."""
    p = compile_predicate("attrs.visibility != 'embargoed'")
    assert p.holds(Item(id="a", attrs={}))
    assert not p.holds(Item(id="b", attrs={"visibility": "embargoed"}))


# ---------------------------------------------------------------------------
# persistence, tenancy, inflate
# ---------------------------------------------------------------------------


def _svc(tmp_path, name="a.db"):
    return EngineService(SqliteStore(tmp_path / name))


def test_state_survives_a_new_service_instance(tmp_path, sim):
    """The point of persistence: a redeploy must not reset users."""
    db = tmp_path / "p.db"
    store = SqliteStore(db)
    svc = EngineService(store)
    svc.register_items("t", sim.items[:60])
    rows = [{"signal_id": f"s{j}", "user_id": "u1", "item_id": it.id,
             "outcome": 1.0 if j % 2 else 0.0, "ts": float(j)}
            for j, it in enumerate(sim.items[:30])]
    svc.observe("t", rows, now=100.0)
    before = svc.profile("t", "u1", now=100.0)
    store.close()

    store2 = SqliteStore(db)
    svc2 = EngineService(store2)
    after = svc2.profile("t", "u1", now=100.0)
    assert after["known"] is True
    assert before["weakest"][0]["tag"] == after["weakest"][0]["tag"]
    assert before["weakest"][0]["level"] == pytest.approx(after["weakest"][0]["level"])
    store2.close()


def test_signals_are_idempotent(tmp_path, sim):
    svc = _svc(tmp_path)
    svc.register_items("t", sim.items[:40])
    rows = [{"signal_id": "fixed", "user_id": "u1", "item_id": sim.items[0].id,
             "outcome": 1.0, "ts": 1.0}]
    first = svc.observe("t", rows, now=10.0)
    second = svc.observe("t", rows, now=10.0)
    assert (first.accepted, first.duplicates) == (1, 0)
    assert (second.accepted, second.duplicates) == (0, 1)


def test_concurrent_observers_lose_no_updates(tmp_path, sim):
    """Two writers, same user, disjoint signals, interleaved small batches on a
    FILE-backed store -- per-thread connections, so both writers really hold
    their own connection. BEGIN IMMEDIATE must serialise the belief
    read-modify-write: every signal lands exactly once, and no 'database is
    locked' escapes to the caller. Raised inside a worker, the lock error
    surfaces through ``f.result()`` and fails the test."""
    svc = EngineService(SqliteStore(tmp_path / "cc.db"))
    svc.register_items("t", sim.items[:40])

    def worker(tag, offset):
        accepted = 0
        for j in range(10):
            rows = [{"signal_id": f"{tag}-{j}-{k}", "user_id": "u1",
                     "item_id": sim.items[offset + j * 2 + k].id,
                     "outcome": 1.0, "ts": float(j)}
                    for k in range(2)]
            accepted += svc.observe("t", rows, now=10.0).accepted
        return accepted

    with ThreadPoolExecutor(2) as ex:
        totals = [f.result() for f in (ex.submit(worker, "a", 0),
                                       ex.submit(worker, "b", 20))]

    assert sum(totals) == 40
    assert svc.store.counts("t")["signals"] == 40
    assert svc.profile("t", "u1", now=10.0)["known"] is True
    svc.store.close()


def test_tenants_cannot_see_each_other(tmp_path, sim):
    svc = _svc(tmp_path)
    svc.register_items("A", sim.items[:50])
    rows = [{"signal_id": f"s{j}", "user_id": "shared_user", "item_id": it.id,
             "outcome": 1.0, "ts": float(j)} for j, it in enumerate(sim.items[:20])]
    svc.observe("A", rows, now=50.0)

    # Same user id, different tenant: must be a stranger with no catalogue.
    assert svc.profile("A", "shared_user", now=50.0)["known"] is True
    prof_b = svc.profile("B", "shared_user", now=50.0)
    assert prof_b.get("known") is False
    dec_b = svc.decide("B", "shared_user", count=5, now=50.0)
    assert dec_b.decision.fallback_reason == "empty_candidate_pool"
    assert svc.store.counts("B")["items"] == 0


def _build_v1_database(path):
    """A database as a v1 deployment left it: only the v1 tables, one item
    carrying tags, version stamped 1. Rows that predate the backfills are the
    whole point -- an upgrade tested on an empty database tests nothing."""
    con = sqlite3.connect(path)
    con.executescript(MIGRATIONS[0][1])
    con.execute("CREATE TABLE schema_version ("
                "version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)")
    con.execute("INSERT INTO schema_version(version, applied_at) VALUES (1, 0.0)")
    con.execute("INSERT INTO items(tenant,item_id,tag_weights,difficulty_prior,attrs) "
                "VALUES ('t','old-item',?,NULL,'{}')",
                (json.dumps({"algebra": 1.0}),))
    con.commit()
    con.close()


def test_store_upgrades_a_v1_database_and_backfills_it(tmp_path):
    """The migrations docstring says the scripts exist 'so the upgrade path is
    exercised rather than assumed' -- but no test ever opened an old database.
    This one does: v1 in, current version out, with both backfills proven
    against a row that predates them."""
    db = tmp_path / "old.db"
    _build_v1_database(db)

    store = SqliteStore(db)
    try:
        assert store.schema_version == MIGRATIONS[-1][0]

        # v2 backfill: the inverted index was built from pre-existing items,
        # or recall would appear to work while returning nothing
        recalled = store.recall_by_tags("t", ["algebra"], 10)
        assert [i.id for i in recalled] == ["old-item"]

        # v5 backfill: the permutation key must be filled in. Left at the
        # default 0, every item shares one position and the coverage slice
        # silently degrades to id order.
        con = sqlite3.connect(db)
        row = con.execute(
            "SELECT shuffle_key FROM items WHERE item_id='old-item'").fetchone()
        con.close()
        assert row[0] != 0

        # and the pre-existing row is still readable through the new schema
        assert store.get_items("t", ["old-item"])[0].tag_weights == {"algebra": 1.0}
        assert store.sample_items("t", 5)[0].id == "old-item"
    finally:
        store.close()


def test_belief_survives_tag_space_growth(tmp_path, sim):
    """A customer adding tags later must not invalidate stored beliefs."""
    svc = _svc(tmp_path)
    svc.register_items("t", sim.items[:20])
    n0 = svc.store.tag_space("t").n_dims
    rows = [{"signal_id": f"s{j}", "user_id": "u1", "item_id": it.id,
             "outcome": 1.0, "ts": float(j)} for j, it in enumerate(sim.items[:10])]
    svc.observe("t", rows, now=10.0)

    svc.register_items("t", sim.items[20:200])         # many new tags
    n1 = svc.store.tag_space("t").n_dims
    assert n1 > n0
    prof = svc.profile("t", "u1", now=20.0)
    assert prof["known"] is True                       # not reset, not crashed


def test_reupserting_an_item_invalidates_its_cached_decomposition(tmp_path):
    """Item metadata and everything derived from it are cached across requests, so
    the generation counter has to expire the derived form too -- otherwise an item
    retagged today keeps being scored against yesterday's dimensions."""
    store = SqliteStore(tmp_path / "memo.db")
    space = store.ensure_tags("t", ["algebra", "geometry"])
    mtor = MTOR(MTORConfig(), TenantItemStore(MTORConfig(), store, "t"))

    store.upsert_items("t", [Item(id="i1", tag_weights={"algebra": 1.0})])
    before = mtor._tag_pairs(store.get_items("t", ["i1"])[0], space)

    store.upsert_items("t", [Item(id="i1", tag_weights={"geometry": 1.0})])
    after = mtor._tag_pairs(store.get_items("t", ["i1"])[0], space)

    assert before == ((space.index_of["algebra"],), (1.0,))
    assert after == ((space.index_of["geometry"],), (1.0,))


def test_get_items_follows_the_requested_order(tmp_path, sim):
    """Explicit-id recall reads from the metadata cache, so the order can no longer
    come from the query; it has to come from the caller, or a replayed decision
    would see its candidates permuted."""
    store = SqliteStore(tmp_path / "order.db")
    store.ensure_tags("t", sorted({t for it in sim.items[:20] for t in it.tag_weights}))
    store.upsert_items("t", sim.items[:20])
    ids = [it.id for it in sim.items[:20]][::-1]
    assert [it.id for it in store.get_items("t", ids)] == ids
    assert [it.id for it in store.get_items("t", ids)] == ids      # warm cache


def test_inflate_raises_variance_with_elapsed_time(tmp_path, sim):
    """Time since last observation must lower confidence, and it must happen on
    read -- there is no background sweep to rely on."""
    svc = _svc(tmp_path)
    svc.register_items("t", sim.items[:40])
    rows = [{"signal_id": f"s{j}", "user_id": "u1", "item_id": it.id,
             "outcome": 1.0, "ts": 0.0} for j, it in enumerate(sim.items[:20])]
    svc.observe("t", rows, now=0.0)

    fresh = svc.profile("t", "u1", now=0.0)
    later = svc.profile("t", "u1", now=0.0 + 400 * 86400.0)
    c_fresh = {r["tag"]: r["confidence"] for r in fresh["weakest"] + fresh["strongest"]}
    c_later = {r["tag"]: r["confidence"] for r in later["weakest"] + later["strongest"]}
    common = set(c_fresh) & set(c_later)
    assert common
    assert any(c_later[t] < c_fresh[t] for t in common)


def test_decision_carries_provenance(tmp_path, sim):
    svc = _svc(tmp_path)
    svc.register_items("t", sim.items[:60])
    r = svc.decide("t", "u1", count=5, goal="practice_weak", now=1.0)
    assert r.decision.model_version
    assert r.decision.policy_id
    # different tune -> different policy id, so audit rows are distinguishable
    r2 = svc.decide("t", "u1", count=5, goal="practice_weak",
                    tune={"difficulty": "brutal"}, now=1.0)
    assert r2.decision.policy_id != r.decision.policy_id


def test_cold_user_is_not_reported_as_high_confidence(tmp_path, sim):
    svc = _svc(tmp_path)
    svc.register_items("t", sim.items[:60])
    r = svc.decide("t", "never_seen", count=5, now=1.0)
    assert r.decision.confidence in ("medium", "low")
    assert r.decision.fallback_reason == "cold_start_no_signals"

