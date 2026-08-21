"""Calibration -- replace guessed constants with measured ones.

The old engine was full of numbers nobody could defend (S=1.84, a 0.3 penalty
weight). ``goals.yaml`` currently carries the same debt one level up: every
value in it is flagged ``uncalibrated``. This module removes the guessing for
the one constant that most directly drives behaviour -- ``rho.target``, the
success probability a learning goal aims a user at -- by measuring it instead of
asserting it.

Method (the honest version of the note in goals.yaml)
-----------------------------------------------------
The claim behind ``peak`` reward is that there is a success-probability band
where practising produces the most learning. That is a testable statement, and
the simulator has ground-truth ability, so we can test it directly:

1. Warm each user to a realistic belief.
2. For a grid of target bands, present the item whose *predicted* success is
   closest to that band, let the user answer, and measure the resulting change
   in true ability on the item's tags (the simulator's _learn/_forget).
3. The band with the greatest mean ability gain is the empirical ``rho.target``.

This is calibration on synthetic data, and it is labelled as such. It does not
license writing a number into a customer SLA -- it licenses replacing "0.70
because it felt right" with "0.70 because on the reference simulator the gain
curve peaks there", plus the curve itself so the next person can disagree with
the evidence rather than the vibe.

The gain curve is also the artefact that tells you whether ``peak`` is even the
right shape: a flat curve would mean the whole peak/target apparatus is
unjustified for this domain, which is exactly the kind of thing that should be
discoverable rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from engine.mtor import MTOR, MTORConfig
from engine.simulator import SimConfig, Simulator


def _warm(sim, mtor, user, n, ts0=0.0):
    b = mtor.init(sim.users[user], sim.space)
    for j, it in enumerate(sim.sample_items(n)):
        b = mtor.update(b, sim.answer(user, it.id, ts0 + j, evolve=False), it)
    return b


def _true_gain_on(sim, user, item_id, before):
    """Change in true ability on the item's tags, summed with tag weights."""
    j = sim.item_index[item_id]
    after = sim.ability[user]
    g = 0.0
    for t, w in sim.item_tags[j].items():
        k = sim.space.index_of[t]
        g += w * (after[k] - before[k])
    return g


def calibrate_rho_target(
    sim: Simulator,
    mtor: MTOR,
    bands: np.ndarray,
    n_users: int = 120,
    warm: int = 40,
    reps_per_band: int = 6,
) -> dict:
    """Return the empirical gain curve over predicted-success bands.

    For each user and band we pick the candidate whose predicted p is nearest
    the band centre, apply it, and record the true ability gain. Selection is by
    the model's own prediction -- the same information the live engine has --
    so the number is achievable in production, not an oracle artefact.
    """
    gains: dict[float, list[float]] = {float(x): [] for x in bands}

    for user in range(min(n_users, sim.cfg.n_users)):
        b = _warm(sim, mtor, user, warm)
        pool = sim.sample_items(120)
        p, _ = mtor.predict(b, pool)

        for band in bands:
            # nearest-prediction items to this band, a few so noise averages out
            order = np.argsort(np.abs(p - band))
            for r in range(reps_per_band):
                it = pool[int(order[r])]
                # Snapshot every piece of state that answer(evolve=True) touches.
                # Resetting ability alone left baseline and last_practice mutated;
                # baseline then accumulated across bands (they run in ascending
                # order), and the next answer's forget-pull toward that inflated
                # baseline biased later bands upward -- manufacturing the very
                # "gain rises to the boundary" shape the curve is meant to test.
                before = sim.ability[user].copy()
                base_before = sim.baseline[user].copy()
                last_before = sim.last_practice[user].copy()
                sim.answer(user, it.id, warm + r + band, evolve=True)
                gains[float(band)].append(_true_gain_on(sim, user, it.id, before))
                sim.ability[user] = before
                sim.baseline[user] = base_before
                sim.last_practice[user] = last_before


    # ddof=1: the population SD estimated with ddof=0 is biased low, which shrinks
    # every SE below and would make a flat curve look separated. The whole point
    # of the separation test is to refuse a spurious argmax, so the error bar has
    # to be the unbiased one.
    def _se(v: list[float]) -> float:
        return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("inf")

    curve = {b: (float(np.mean(v)), _se(v)) for b, v in gains.items()}
    best = max(curve, key=lambda b: curve[b][0])

    # Is the peak real? Compare best band to the flattest alternative via the
    # separation in standard errors. A peak within 1 SE of its neighbours is not
    # a peak, and we say so rather than reporting a spurious argmax.
    means = np.array([curve[b][0] for b in bands])
    ses = np.array([curve[b][1] for b in bands])
    spread = float(means.max() - means.min())
    pooled_se = float(np.sqrt((ses ** 2).mean()))
    separated = spread > 2.0 * pooled_se

    # An argmax sitting on the edge of the grid is not an interior optimum; it
    # means the data says "keep going in that direction" and the grid ran out.
    # Reporting it as a calibrated target would dress a monotone relationship up
    # as a peak, which is precisely the sort of laundering this module exists to
    # stop.
    at_edge = bool(np.argmax(means) in (0, len(means) - 1))
    dif = np.diff(means)
    monotone = bool(np.all(dif >= -pooled_se) or np.all(dif <= pooled_se))

    return {
        "curve": {round(b, 3): {"gain": round(curve[b][0], 5),
                                "se": round(curve[b][1], 5)} for b in bands},
        "best_target": round(float(best), 3),
        "peak_separated": bool(separated),
        "argmax_at_grid_edge": at_edge,
        "monotone": monotone,
        "interior_peak": bool(separated and not at_edge and not monotone),
        "spread": round(spread, 5),
        "pooled_se": round(pooled_se, 5),
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="write JSON report here")
    ap.add_argument("--users", type=int, default=120)
    ap.add_argument("--warm", type=int, default=40)
    args = ap.parse_args()

    if args.users < 1 or args.warm < 1:
        # With no users every band's sample list is empty, so the curve is all
        # NaN and the report would be a table of nothing dressed as a result.
        ap.error("--users and --warm must both be >= 1")


    sim = Simulator(SimConfig())
    mtor = MTOR(MTORConfig(floor_attr="floor"))
    bands = np.round(np.arange(0.30, 0.91, 0.10), 2)

    print("calibrating rho.target (gain vs predicted-success band):")
    rep = calibrate_rho_target(sim, mtor, bands, n_users=args.users, warm=args.warm)
    for b, cell in rep["curve"].items():
        bar = "#" * max(0, int(cell["gain"] * 400))
        print(f"  p~{b:.2f}: gain={cell['gain']:+.4f} +/-{cell['se']:.4f}  {bar}")
    print(f"  best_target={rep['best_target']}  peak_separated={rep['peak_separated']}"
          f"  interior_peak={rep['interior_peak']}"
          f"  (spread {rep['spread']:.4f} vs pooled_se {rep['pooled_se']:.4f})")

    if not rep["peak_separated"]:
        print("  VERDICT: gain curve is flat within 2 pooled SE -- the peak reward\n"
              "           shape is not justified by this data at all.")
    elif not rep["interior_peak"]:
        print("  VERDICT: gain rises monotonically to the grid edge, so there is NO\n"
              "           interior optimum. On this reference simulator, 'easier is\n"
              "           always better' -- its learn rule is gain ~ (0.4 + 0.6*outcome)\n"
              "           with no desirable-difficulty term, so it structurally cannot\n"
              "           produce a peak. Adding one would only recover whatever was\n"
              "           injected, which is circular.\n"
              "           => rho.target stays UNCALIBRATED. What is validated here is\n"
              "              the estimator, not the value: it correctly refuses to\n"
              "              report a peak that the data does not contain. A real\n"
              "              rho.target requires logged interactions with a genuine\n"
              "              state-gain signal.")
    else:
        print(f"  VERDICT: interior peak at {rep['best_target']} is separated from the\n"
              f"           grid edges; this value is empirically supported on this data.")


    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
