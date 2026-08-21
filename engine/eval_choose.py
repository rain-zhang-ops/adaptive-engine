"""End-to-end Choose validation on the simulator.

Everything above this point was measured in isolation -- MTOR's calibration,
the scorer's utility terms, the policy translation. This harness runs the whole
chain the way a customer would call it,

    goal (+tune) --policy--> Utility
    signals      --MTOR-----> Belief
    Belief,items --scorer----> Scored
    Scored       --chooser---> Decision

and checks the properties that only exist once the pieces are composed and that
a customer would actually notice if broken:

* hard constraints are never violated (predicate / exclude / quota / max_per_tag)
* the SAME belief under 'broad' vs 'narrow' focus produces measurably more vs
  less tag-diverse sets -- i.e. the structure term actually reverses direction,
  not just changes a number
* explore slots carry a propensity that matches their true marginal inclusion
  probability, verified by Monte-Carlo over many seeds -- because a wrong
  propensity is an invisible bias, not a crash
* the goal-vs-goal contrast that is the product's whole pitch: practice_weak
  spends its picks on low-mu (weak) dimensions, more_like_this on high-mu
  (strong) ones, from one identical belief

No thresholds are invented. Diversity is compared between two policies on the
same data (a relative claim), and propensity is compared to its own analytic
value (an exact claim).
"""

from __future__ import annotations

import math
import sys
from collections import Counter

import numpy as np

from contracts.core import Constraints, Quota
from engine.chooser import GreedyChooser
from engine.mtor import MTOR, MTORConfig
from engine.policy import constraints_from, load_catalog
from engine.scorer import UtilityScorer
from engine.simulator import SimConfig, Simulator


def _warm_belief(sim, mtor, scorer, user, n_signals, ts0=0.0):
    """Feed a user some interactions so the belief has structure to act on."""
    b = mtor.init(sim.users[user], sim.space)
    items = sim.sample_items(n_signals)
    for j, it in enumerate(items):
        sig = sim.answer(user, it.id, ts0 + j, evolve=False)
        b = mtor.update(b, sig, it)
    return b


def _tag_diversity(items):
    """Number of distinct tags touched by the set. Coarse but monotone: a more
    diverse set touches more tags, and it needs no free parameter to state."""
    tags = set()
    for it in items:
        tags.update(it.tag_weights)
    return len(tags)


def _dominant_tag_share(items):
    c = Counter()
    for it in items:
        for t, w in it.tag_weights.items():
            c[t] += w
    total = sum(c.values())
    return max(c.values()) / total if total else 0.0


# ---------------------------------------------------------------------------


def test_hard_constraints(sim, mtor, scorer):
    cat = load_catalog()
    chooser = GreedyChooser(seed=1)
    u = cat.utility("practice_weak")

    b = _warm_belief(sim, mtor, scorer, user=0, n_signals=40)
    pool = sim.sample_items(120)

    embargo = frozenset(it.id for it in pool[:10])
    constraints = Constraints(
        k=8,
        predicates=("attrs.kind != 'essay'",),
        quotas=(Quota(group_by="attrs.kind", counts={"choice": 3}),),
        max_per_tag=2,
        exclude_item_ids=embargo,
    )

    scored = scorer.value(b, pool, u)
    dec = chooser.solve(scored, pool, u, constraints)
    chosen_items = [it for it in pool if it.id in {c.item_id for c in dec.chosen}]

    assert dec.confidence == "high", f"expected high confidence, got {dec.fallback_reason}"
    assert len(dec.chosen) == 8, f"k=8 not met: {len(dec.chosen)}"
    assert not ({c.item_id for c in dec.chosen} & embargo), "embargoed item leaked"
    assert all(it.attrs["kind"] != "essay" for it in chosen_items), "predicate violated"

    kinds = Counter(it.attrs["kind"] for it in chosen_items)
    assert kinds["choice"] >= 3, f"quota choice>=3 not met: {kinds}"

    tag_use = Counter()
    for it in chosen_items:
        for t in it.tag_weights:
            tag_use[t] += 1
    assert max(tag_use.values()) <= 2, f"max_per_tag=2 violated: {tag_use.most_common(3)}"
    print(f"  constraints: k={len(dec.chosen)} kinds={dict(kinds)} "
          f"max_tag={max(tag_use.values())} embargo_clean=True  OK")


def test_infeasible_degrades(sim, mtor, scorer):
    """An unsatisfiable quota must degrade to low confidence, never raise:
    a 5xx from the engine is an outage in the caller's product."""
    cat = load_catalog()
    chooser = GreedyChooser(seed=1)
    u = cat.utility("practice_weak")
    b = _warm_belief(sim, mtor, scorer, user=0, n_signals=20)
    pool = [it for it in sim.sample_items(60) if it.attrs["kind"] == "choice"][:20]

    constraints = Constraints(k=5, quotas=(Quota("attrs.kind", {"essay": 3}),))
    scored = scorer.value(b, pool, u)
    dec = chooser.solve(scored, pool, u, constraints)
    assert dec.confidence == "low", "infeasible quota should degrade"
    assert dec.fallback_reason == "constraints_unsatisfiable", dec.fallback_reason
    print(f"  infeasible: confidence={dec.confidence} reason={dec.fallback_reason}  OK")


def test_structure_reversal(sim, mtor, scorer):
    """Same belief, same candidates, same goal: only the 'focus' tune knob
    differs. Holding the goal fixed means the diversity gap isolates focus
    itself, not a confounded swap of gamma/value/rho across two goals.
    This is a relative claim between two policies, so it needs no absolute
    diversity threshold."""
    cat = load_catalog()
    b = _warm_belief(sim, mtor, scorer, user=3, n_signals=50)
    pool = sim.sample_items(150)

    u_broad = cat.utility("practice_weak", {"focus": "broad"})
    u_narrow = cat.utility("practice_weak", {"focus": "narrow"})

    div_broad, div_narrow = [], []
    share_broad, share_narrow = [], []
    for seed in range(12):
        ch = GreedyChooser(seed=seed)
        cons = Constraints(k=8)
        for u, dstore, sstore in ((u_broad, div_broad, share_broad),
                                   (u_narrow, div_narrow, share_narrow)):
            dec = ch.solve(scorer.value(b, pool, u), pool, u, cons)
            picked = [it for it in pool if it.id in {c.item_id for c in dec.chosen}]
            dstore.append(_tag_diversity(picked))
            sstore.append(_dominant_tag_share(picked))

    mb, mn = float(np.mean(div_broad)), float(np.mean(div_narrow))
    sb, sn = float(np.mean(share_broad)), float(np.mean(share_narrow))
    assert mb > mn, f"broad ({mb:.1f}) not more diverse than narrow ({mn:.1f})"
    assert sn > sb, f"narrow dominant-share ({sn:.2f}) not > broad ({sb:.2f})"
    print(f"  structure: broad tags={mb:.1f} share={sb:.2f} | "
          f"narrow tags={mn:.1f} share={sn:.2f}  OK (diverge as designed)")


def test_propensity_matches_frequency(sim, mtor, scorer):
    """The reported propensity must equal the empirical selection frequency of
    explore items over many seeds. A wrong propensity is a silent IPS bias, so
    it is checked against its own Monte-Carlo estimate, not a guessed tolerance."""
    cat = load_catalog()
    u = cat.utility("explore")            # explore_floor 0.30 -> guaranteed explore slots
    b = _warm_belief(sim, mtor, scorer, user=7, n_signals=40)
    pool = sim.sample_items(60)
    scored = scorer.value(b, pool, u)
    cons = Constraints(k=8)

    n_trials = 4000
    counts = Counter()
    reported = {}
    exact_seen = 0
    total_explore = 0
    for seed in range(n_trials):
        dec = GreedyChooser(seed=seed).solve(scored, pool, u, cons)
        for c in dec.chosen:
            if c.reasons.get("explore", 0.0) == 1.0:
                counts[c.item_id] += 1
                reported[c.item_id] = c.propensity
                total_explore += 1
                exact_seen += int(c.reasons.get("propensity_exact", 0.0))

    # Verify against items that were explored often enough for a stable rate.
    checked = 0
    for item_id, cnt in counts.items():
        if cnt < 200:
            continue
        emp = cnt / n_trials
        rep = reported[item_id]
        # Binomial SE of the empirical rate. Several items are each tested at
        # 4 sigma, so the per-item bound is deliberately loose to keep the
        # family-wise false-positive rate low across the batch (4 sigma per test
        # is ~6e-5 one-sided). No absolute slack is added: for every item that
        # clears the cnt>=200 gate, emp>=0.05 and 4*SE already exceeds 0.013, so
        # a fixed floor would only ever weaken a bound that is tight enough.
        se = math.sqrt(max(emp * (1 - emp), 1e-9) / n_trials)
        assert abs(emp - rep) <= 4 * se, (
            f"item {item_id}: reported propensity {rep:.3f} vs empirical {emp:.3f} "
            f"(4SE={4*se:.3f})")
        checked += 1

    assert checked >= 3, f"too few explore items to validate ({checked})"
    frac_exact = exact_seen / max(total_explore, 1)
    print(f"  propensity: {checked} items match empirical within 4SE; "
          f"exact-flag on {frac_exact:.0%} of explore picks  OK")


def test_goal_intent_separates(sim, mtor, scorer):
    """The product pitch in one assertion: from ONE belief, practice_weak must
    concentrate its picks on the user's WEAK dimensions and more_like_this on the
    STRONG ones. Measured as the belief-mu of the tags each policy selects."""
    cat = load_catalog()
    # A user with a deliberately lopsided profile so weak/strong is unambiguous.
    b = _warm_belief(sim, mtor, scorer, user=11, n_signals=80)

    pool = sim.sample_items(150)
    idx = b.space.index_of

    def mean_selected_mu(goal):
        u = cat.utility(goal)
        dec = GreedyChooser(seed=0).solve(scorer.value(b, pool, u), pool, u,
                                          Constraints(k=10))
        vals = []
        for c in dec.chosen:
            if c.reasons.get("explore", 0.0) == 1.0:
                continue                      # exploit picks reflect the intent
            it = sim.item_by_id(c.item_id)
            for t, w in it.tag_weights.items():
                vals.append(w * b.mu[idx[t]])
        return float(np.mean(vals)) if vals else 0.0

    weak = mean_selected_mu("practice_weak")
    strong = mean_selected_mu("more_like_this")
    assert strong > weak, (
        f"more_like_this ({strong:+.3f}) should target higher-mu tags than "
        f"practice_weak ({weak:+.3f})")
    print(f"  intent: practice_weak mu={weak:+.3f} < more_like_this mu={strong:+.3f}  "
          f"OK (remediation vs amplification separate)")


def main():
    sim = Simulator(SimConfig())
    mtor = MTOR(MTORConfig(floor_attr="floor"))
    scorer = UtilityScorer(mtor)

    print("end-to-end Choose:")
    test_hard_constraints(sim, mtor, scorer)
    test_infeasible_degrades(sim, mtor, scorer)
    test_structure_reversal(sim, mtor, scorer)
    test_propensity_matches_frequency(sim, mtor, scorer)
    test_goal_intent_separates(sim, mtor, scorer)
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
