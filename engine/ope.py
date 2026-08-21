"""Off-policy evaluation -- turning logged propensities into an answer.

Recording a propensity is only half the job; something has to consume it. This
module is that consumer, and it is what makes ``gamma`` / ``Phi`` calibration
possible at all: those weights cannot be tuned on synthetic data (the dynamics
would be my own assumption fed back to me), so they have to be searched against
logged traffic -- which requires estimating "what would policy B have earned?"
from data collected under policy A.

Estimator
---------
Because the chooser reports *marginal* inclusion probabilities, the slate value
is exactly IPS-estimable at the item level:

    V(pi_t) = E_l [ sum_{a in A_l} (pi_t(a) / pi_l(a)) * r_a ]

which is unbiased for E_t[ sum_{a in A_t} r_a ]. This identity is the reason the
propensity semantics had to be marginal rather than per-draw -- with the
conditional probability the same expression is off by a factor of the exploration
budget.

What is reported alongside the number, and why
----------------------------------------------
A bare IPS point estimate is close to useless, because it fails silently in two
ways this module makes visible:

``coverage``  share of the target policy's total inclusion mass that appears in
              the logged rows. It bounds how much of the estimate rests on
              observed data rather than on the weighting. Note what it is *not*:
              with a uniform logging policy sampling k of N candidates, coverage
              is ~k/N by construction and that is not a defect -- it is the price
              of a sparse log. What would be a defect is target mass sitting on
              items the logging policy could never have shown, which shows up as
              coverage far below the sampling rate.
``ess``       effective sample size, (sum w)^2 / sum w^2. A handful of enormous
              weights can produce a confident-looking mean built from three rows.
``clipped``   fraction of weights hitting the clip. Clipping trades bias for
              variance; hiding how much of it happened hides the bias.


Running this file validates the estimator against ground truth on the simulator:
log under a uniform policy, estimate a target policy's value off-policy, then run
that target policy on-policy and compare. The estimator is only trustworthy on
real logs if it can recover a known answer here first.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from contracts.core import Constraints
from engine.chooser import GreedyChooser
from engine.mtor import MTOR, MTORConfig
from engine.policy import load_catalog
from engine.scorer import UtilityScorer
from engine.simulator import SimConfig, Simulator

__all__ = ["LoggedItem", "OPEResult", "evaluate"]


@dataclass(frozen=True)
class LoggedItem:
    reward: float
    logged_propensity: float
    target_prob: float
    decision_id: int


@dataclass(frozen=True)
class OPEResult:
    n_items: int
    n_decisions: int
    slate_value_ips: float
    slate_value_se: float
    per_item_snips: float
    ess: float
    support: float
    clipped_fraction: float
    dropped_no_propensity: int = 0
    support_known: bool = True

    def report(self) -> str:
        support = f"{self.support:.2%}" if self.support_known else "unknown"
        extra = (f"  dropped_no_propensity={self.dropped_no_propensity}"
                 if self.dropped_no_propensity else "")
        return (f"slate_value_ips={self.slate_value_ips:.4f} "
                f"+/-{self.slate_value_se:.4f}  per_item_snips={self.per_item_snips:.4f}  "
                f"ess={self.ess:.1f}/{self.n_items}  support={support}  "
                f"clipped={self.clipped_fraction:.2%}{extra}")


def evaluate(logged: Sequence[LoggedItem], clip: float = 20.0,
             target_mass: float | None = None) -> OPEResult:
    if not logged:
        return OPEResult(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         dropped_no_propensity=0, support_known=target_mass is not None)

    # IPS requires the logging policy to be positive on the target's support. A
    # row with propensity <= 0 is not "very surprising", it is *unusable*:
    # coercing it to 1e-12 turns it into an astronomically large weight that
    # clipping then caps at ``clip``, quietly biasing the estimate instead of
    # excluding the row. Drop and count them.
    usable = [l for l in logged if l.logged_propensity > 0.0]
    dropped = len(logged) - len(usable)
    if not usable:
        return OPEResult(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         dropped_no_propensity=dropped,
                         support_known=target_mass is not None)

    w_raw = np.array([l.target_prob / l.logged_propensity for l in usable])
    r = np.array([l.reward for l in usable])
    w = np.minimum(w_raw, clip)
    clipped = float((w_raw > clip).mean())

    dec_ids = np.array([l.decision_id for l in usable])
    n_dec = int(len(np.unique(dec_ids)))

    # Slate value: sum weighted reward within a decision, then average across
    # decisions. Averaging per item instead would silently rescale by slate size.
    per_dec: dict[int, float] = {}
    for wi, ri, d in zip(w, r, dec_ids):
        per_dec[int(d)] = per_dec.get(int(d), 0.0) + float(wi * ri)
    vals = np.array(list(per_dec.values()))
    value = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0

    sw = float(w.sum())
    snips = float((w * r).sum() / sw) if sw > 0 else 0.0
    ess = float(sw ** 2 / float((w ** 2).sum())) if sw > 0 else 0.0

    # Coverage is only meaningful against a known target mass. Defaulting a
    # missing denominator to "100% covered" reports full support precisely when
    # the caller forgot to supply the one number that would refute it.
    if target_mass:
        covered = float(sum(l.target_prob for l in usable))
        support = min(covered / target_mass, 1.0)
        support_known = True
    else:
        support = 0.0
        support_known = False

    return OPEResult(n_items=len(usable), n_decisions=n_dec, slate_value_ips=value,
                     slate_value_se=se, per_item_snips=snips, ess=ess,
                     support=support, clipped_fraction=clipped,
                     dropped_no_propensity=dropped, support_known=support_known)



# ---------------------------------------------------------------------------
# validation against ground truth
# ---------------------------------------------------------------------------


def _warm(sim, mtor, user, n):
    b = mtor.init(sim.users[user], sim.space)
    for j, it in enumerate(sim.sample_items(n)):
        b = mtor.update(b, sim.answer(user, it.id, j, evolve=False), it)
    return b


def validate(n_users: int = 150, k: int = 8, warm: int = 40,
             target_goal: str = "challenge") -> dict:
    """Log under a uniform policy, estimate the target off-policy, then run the
    target for real and compare.

    A uniform logging policy is used on purpose: it is what a bootstrap logging
    phase looks like, and it gives full support so the estimator is being tested
    rather than the overlap.
    """
    sim = Simulator(SimConfig())
    mtor = MTOR(MTORConfig(floor_attr="floor"))
    scorer = UtilityScorer(mtor)
    cat = load_catalog()
    utility = cat.utility(target_goal)
    cons = Constraints(k=k)
    rng = np.random.default_rng(7)

    logged: list[LoggedItem] = []
    target_mass = 0.0
    on_policy = []

    for user in range(min(n_users, sim.cfg.n_users)):
        belief = _warm(sim, mtor, user, warm)
        pool = sim.sample_items(80)
        scored = scorer.value(belief, pool, utility)

        # target policy's exact marginal inclusion probabilities
        probs = GreedyChooser(seed=0).inclusion_probabilities(scored, pool, utility, cons)
        target_mass += float(sum(probs.values()))

        # -- logging policy: uniform k-subset of the pool ------------------
        idx = rng.choice(len(pool), size=k, replace=False)
        prop = k / len(pool)
        for j in idx:
            it = pool[int(j)]
            outcome = 1.0 if rng.random() < sim.p_true(user, it.id) else 0.0
            logged.append(LoggedItem(reward=outcome, logged_propensity=prop,
                                     target_prob=probs[it.id], decision_id=user))

        # -- ground truth: what the target policy actually earns -----------
        # E_t[sum_{a in A_t} r_a] = sum_a pi_t(a) * p_true(a). Using the exact
        # expectation rather than one sampled slate removes the exploration
        # noise from the reference, so any gap that shows up is the estimator's.
        on_policy.append(sum(probs[it.id] * sim.p_true(user, it.id) for it in pool))


    res = evaluate(logged, target_mass=target_mass)
    truth = float(np.mean(on_policy))
    truth_se = float(np.std(on_policy, ddof=1) / math.sqrt(len(on_policy)))
    return {"ope": res, "truth": truth, "truth_se": truth_se}


def main() -> int:
    print("off-policy evaluation -- estimator validated against on-policy truth")
    goals = ("challenge", "more_like_this", "screen")
    # Three goals means three tests. The 3-sigma per-test band is kept
    # (family-wise false-positive ~0.8% under the normal approximation), but the
    # "OK" below is a sanity gate, not a proof of unbiasedness: it only says the
    # gap is small relative to *this run's* noise. A single run passing is
    # necessary, not sufficient -- re-run across seeds before trusting a pass.
    mismatches = 0
    for goal in goals:
        out = validate(target_goal=goal)
        res: OPEResult = out["ope"]
        truth, tse = out["truth"], out["truth_se"]
        gap = res.slate_value_ips - truth
        pooled = math.sqrt(res.slate_value_se ** 2 + tse ** 2)
        verdict = "OK" if abs(gap) <= 3.0 * pooled else "MISMATCH"
        if verdict != "OK":
            mismatches += 1
        print(f"\n  target={goal}")
        print(f"    {res.report()}")
        print(f"    on-policy truth={truth:.4f} +/-{tse:.4f}")
        print(f"    gap={gap:+.4f}  (3 pooled SE = {3*pooled:.4f})  -> {verdict}")
        if verdict != "OK":
            print("    NOTE: a gap beyond 3 SE means the estimator, the propensity "
                  "semantics, or the support is wrong -- not that the policy is bad.")
    print("\nHow to read: the estimator is only usable on real logs if it recovers "
          "a known\nanswer here. support/ess are part of the result, not footnotes.")
    print(f"\n{len(goals) - mismatches}/{len(goals)} goals within 3 pooled SE on "
          "this run; the band is the run's own noise, not a cross-seed guarantee.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
