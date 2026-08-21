"""Evaluation harness for MTOR. Synthetic data only; numpy is the sole dependency.

Design principle: never score against an invented threshold. Every metric is
reported next to the **oracle** -- the score obtained by a model that knows the
simulator's true probabilities. The oracle is not 1.0: a guessing floor on
multiple-choice items injects irreducible noise, so "AUC 0.68" is meaningless
until you know whether the ceiling is 0.70 or 0.95. What matters is the gap.

Metrics
-------
1. ECE -- calibration. More decision-relevant than AUC here, because items are
   chosen by the absolute value of p_hat; correct ranking with biased magnitudes
   still selects the wrong difficulty. This is the number a calibration SLA is
   written against.
2. AUC / Brier -- discrimination, reported against the oracle ceiling.
3. Ability recovery -- Pearson and Spearman between mu and hidden ground truth.
   Pearson is depressed by link/scale mismatch (MTOR is Rasch, the generator is
   3PL); Spearman isolates ordering, which is what selection actually consumes.
4. CAT convergence -- measured as prediction error on a **fixed probe set**
   held out before administration starts. Using latent-ability RMSE over the
   growing "dimensions observed so far" set is confounded: the comparison set
   changes as more items are administered, so the number can rise even while
   the estimate improves.

Guardrails
----------
 - Elo baseline: if a point-estimate rating matches MTOR, the variance
   machinery is not earning its keep.
 - Untagged path: with tags stripped the engine must still run and extract
   signal, since requiring a taxonomy up front is an adoption blocker.

Run:  python engine/eval_mtor.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.core import Item, Signal, TagSpace  # noqa: E402

from engine.mtor import MTOR, ItemStore, MTORConfig  # noqa: E402
from engine.simulator import SimConfig, Simulator  # noqa: E402

DAY = 86400.0


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi) if hi >= 1.0 else (p >= lo) & (p < hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(p, y, n_bins: int = 10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi) if hi >= 1.0 else (p >= lo) & (p < hi)
        if m.any():
            rows.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def auc(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + csum + 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def spearman(a, b) -> float:
    def rank(x):
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------
# Elo baseline: point estimate, no variance
# --------------------------------------------------------------------------

class EloBaseline:
    def __init__(self, space: TagSpace, k: float = 0.1, spread: float = 2.0):
        self.space = space
        self.k = k
        self.spread = spread
        self.r: dict[str, np.ndarray] = {}
        self.b: dict[str, float] = {}

    def _u(self, uid: str) -> np.ndarray:
        return self.r.setdefault(uid, np.zeros(self.space.n_dims))

    def _bias(self, item: Item) -> float:
        if item.id not in self.b:
            d = 0.0 if item.difficulty_prior is None else (2 * item.difficulty_prior - 1) * self.spread
            self.b[item.id] = d
        return self.b[item.id]

    def _wv(self, item: Item):
        pairs = [(self.space.index_of[t], w) for t, w in item.tag_weights.items()
                 if t in self.space.index_of and w > 0]
        if not pairs:
            return np.empty(0, dtype=int), np.empty(0)
        idx = np.array([i for i, _ in pairs], dtype=int)
        a = np.array([w for _, w in pairs], dtype=float)
        a /= a.sum()
        return idx, a

    def predict_one(self, uid: str, item: Item) -> float:
        idx, a = self._wv(item)
        if idx.size == 0:
            return 0.5
        theta = float(np.dot(a, self._u(uid)[idx]))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, theta - self._bias(item)))))

    def update(self, uid: str, item: Item, outcome: float) -> None:
        idx, a = self._wv(item)
        if idx.size == 0:
            return
        err = outcome - self.predict_one(uid, item)
        self._u(uid)[idx] += self.k * a * err
        self.b[item.id] -= self.k * err


# --------------------------------------------------------------------------
# [1] prequential online evaluation, with oracle ceiling
# --------------------------------------------------------------------------

def run_prequential(seed_shift: int = 0, steps: int = 60, warmup: int = 5):
    """Test-then-train: predict each interaction before its outcome is revealed.

    Runs MTOR twice -- with and without the baseline-success term -- because the
    reliability table of the plain Rasch variant shows a textbook unmodelled-floor
    signature (low bins under-predict, high bins unbiased). Naming the ablation
    here is what turns "the number is mediocre" into "this specific missing term
    costs this much".
    """
    sim = Simulator(SimConfig(seed=20260820 + seed_shift))

    variants = {
        name: MTOR(cfg, ItemStore(cfg))
        for name, cfg in (
            ("mtor_no_floor", MTORConfig()),
            ("mtor_floor", MTORConfig(floor_attr="floor")),
            ("mtor_floor_disc", MTORConfig(floor_attr="floor", learn_discrimination=True,
                                           disc_min_exposure=10)),
        )
    }


    beliefs = {name: {u: m.init(u, sim.space) for u in sim.users} for name, m in variants.items()}
    elo = EloBaseline(sim.space)

    preds: dict[str, list[float]] = {name: [] for name in variants}
    preds["elo"] = []
    preds["oracle"] = []
    Y: list[float] = []

    for step in range(steps):
        ts = (step + 1) * DAY
        for ui, uid in enumerate(sim.users):
            item = sim.sample_items(1)[0]

            staged = {}
            for name, m in variants.items():
                b = m.inflate(beliefs[name][uid], ts)
                staged[name] = b
                if step >= warmup:
                    preds[name].append(float(m.predict(b, [item])[0][0]))

            p_e = elo.predict_one(uid, item)
            p_o = sim.p_true(ui, item.id)

            sig = sim.answer(ui, item.id, ts, evolve=True)
            if step >= warmup:
                preds["elo"].append(p_e)
                preds["oracle"].append(p_o)
                Y.append(sig.outcome)

            for name, m in variants.items():
                beliefs[name][uid] = m.update(staged[name], sig, item)
            elo.update(uid, item, sig.outcome)

    out = {}
    for name, P in preds.items():
        out[name] = {"ece": expected_calibration_error(P, Y), "auc": auc(P, Y), "brier": brier(P, Y)}
    out["_n"] = len(Y)
    out["_rel_no_floor"] = reliability_table(preds["mtor_no_floor"], Y)
    out["_rel_floor"] = reliability_table(preds["mtor_floor"], Y)
    return out



# --------------------------------------------------------------------------
# [2] ability recovery
# --------------------------------------------------------------------------

def run_ability_recovery(n_users: int = 40, n_obs: int = 200):
    """Ability frozen so there is a fixed target to recover."""
    sim = Simulator(SimConfig(seed=7))
    cfg = MTORConfig(floor_attr="floor")
    mtor = MTOR(cfg, ItemStore(cfg))

    rng = np.random.default_rng(99)
    picks = rng.choice(len(sim.users), size=n_users, replace=False)

    pear, spear = [], []
    for ui in picks:
        uid = sim.users[int(ui)]
        b = mtor.init(uid, sim.space)
        for step in range(n_obs):
            item = sim.sample_items(1)[0]
            b = mtor.update(b, sim.answer(int(ui), item.id, (step + 1) * DAY, evolve=False), item)
        truth = sim.true_ability(int(ui))
        seen = ~np.isnan(b.last_seen)

        if seen.sum() >= 4:
            pear.append(float(np.corrcoef(b.mu[seen], truth[seen])[0, 1]))
            spear.append(spearman(b.mu[seen], truth[seen]))
    return {
        "pearson_mean": float(np.mean(pear)),
        "pearson_min": float(np.min(pear)),
        "spearman_mean": float(np.mean(spear)),
        "spearman_min": float(np.min(spear)),
        "note": "Pearson is depressed by Rasch-vs-3PL link mismatch; Spearman "
                "isolates ordering, which is what selection consumes.",
    }


# --------------------------------------------------------------------------
# [3] CAT convergence on a fixed held-out probe set
# --------------------------------------------------------------------------

def run_cat_convergence(budget: int = 30, n_users: int = 30, probe_size: int = 60):
    """Measure prediction error against oracle probabilities on a probe set that
    is fixed before administration begins.

    Two traps avoided here:

    1. Latent-ability RMSE over "dimensions observed so far" is confounded --
       the comparison set grows as items are administered, so the number can
       rise even while the estimate improves. A fixed probe set removes that.
    2. The item pool and probe set must be *identical* across strategies for a
       given user. Drawing them from a shared, advancing RNG gives each strategy
       a different pool, and then any difference in the curves is partly just
       different draws. Pools are therefore built from a per-user seed that does
       not depend on which strategy runs first.
    3. So must the outcomes. Aligning only the pools still left both strategies
       drawing coin flips from the simulator's shared RNG, which advances -- so
       the second strategy to run met a different noise sequence. Outcomes are
       therefore pinned to (user, item) via ``coins_for``, which is what makes
       the step-0 assertion below more than a formality.
    """

    sim = Simulator(SimConfig(seed=42))
    users = np.random.default_rng(5).choice(len(sim.users), size=n_users, replace=False)

    def build_pool(ui: int):
        """Deterministic per-user pool/probe split, strategy-independent."""
        rng = np.random.default_rng(90000 + ui)
        picks = rng.choice(len(sim.items), size=probe_size + 80, replace=False)
        pool = [sim.items[int(p)] for p in picks]
        return pool[:probe_size], pool[probe_size:]

    def probe_error(mtor: MTOR, belief, ui: int, probe: list[Item]) -> float:
        p_hat, _ = mtor.predict(belief, probe)
        p_true = np.array([sim.p_true(ui, it.id) for it in probe])
        return float(np.sqrt(np.mean((p_hat - p_true) ** 2)))

    def coins_for(ui: int) -> np.ndarray:
        """One pre-drawn uniform per item, per user -- common random numbers.

        ``sim.answer`` draws its outcome from the simulator's shared RNG, which
        advances. Running one strategy to completion before the other therefore
        left the second facing a different coin sequence, so part of the gap
        between the curves was outcome noise rather than selection quality.
        Pinning the coin to (user, item) removes that: whichever strategy reaches
        an item, it gets the same outcome. Valid here because ``evolve=False``
        freezes ability, so ``p_true`` does not depend on what came before.
        """
        return np.random.default_rng(50000 + ui).random(len(sim.items))

    def paired_answer(ui: int, item_id: str, ts: float, coins: np.ndarray) -> Signal:
        p = sim.p_true(ui, item_id)
        outcome = 1.0 if coins[sim.item_index[item_id]] < p else 0.0
        return Signal(user_id=sim.users[ui], item_id=item_id, outcome=outcome, ts=ts)

    def run(strategy: str) -> np.ndarray:
        curves = []
        for ui in users:
            ui = int(ui)
            uid = sim.users[ui]
            cfg = MTORConfig(floor_attr="floor")
            mtor = MTOR(cfg, ItemStore(cfg))
            b = mtor.init(uid, sim.space)

            probe, admin_pool = build_pool(ui)
            # independent selection randomness, also per-user so both strategies
            # face the same noise budget
            pick_rng = np.random.default_rng(70000 + ui)
            coins = coins_for(ui)
            used: set[str] = set()

            errs = [probe_error(mtor, b, ui, probe)]
            for t in range(budget):
                avail = [it for it in admin_pool if it.id not in used]
                if not avail:
                    break
                if strategy == "cat":
                    pick = avail[int(np.argmax(mtor.fisher_information(b, avail)))]
                else:
                    pick = avail[int(pick_rng.integers(len(avail)))]
                used.add(pick.id)
                b = mtor.update(
                    b, paired_answer(ui, pick.id, (t + 1) * DAY, coins), pick)
                errs.append(probe_error(mtor, b, ui, probe))
            curves.append(errs)
        return np.mean(np.array(curves), axis=0)


    cat, rnd = run("cat"), run("random")
    assert abs(cat[0] - rnd[0]) < 1e-9, "step-0 error must match; pools are not aligned"

    target = rnd[-1]
    reached = next((i for i, e in enumerate(cat) if e <= target), budget)
    return {
        "probe_rmse_cat": {0: float(cat[0]), 5: float(cat[5]), 10: float(cat[10]),
                           20: float(cat[20]), budget: float(cat[budget])},
        "probe_rmse_random": {0: float(rnd[0]), 5: float(rnd[5]), 10: float(rnd[10]),
                              20: float(rnd[20]), budget: float(rnd[budget])},
        "cat_items_to_match_random_final": reached,
        "reduction_cat": float(cat[0] - cat[budget]),
        "reduction_random": float(rnd[0] - rnd[budget]),
        "budget": budget,
        "note": "lower is better; index 0 is the shared prior, before any observation",
    }



# --------------------------------------------------------------------------
# [4] untagged degradation
# --------------------------------------------------------------------------

def run_untagged(steps: int = 40, warmup: int = 5):
    sim = Simulator(SimConfig(seed=11))
    space = TagSpace(index_of={TagSpace.LATENT: 0}, tag_of=[TagSpace.LATENT])
    cfg = MTORConfig(floor_attr="floor")
    mtor = MTOR(cfg, ItemStore(cfg))
    beliefs = {u: mtor.init(u, space) for u in sim.users}

    P, P_o, Y = [], [], []
    for step in range(steps):
        ts = (step + 1) * DAY
        for ui, uid in enumerate(sim.users):
            raw = sim.sample_items(1)[0]
            item = Item(id=raw.id, tag_weights={}, difficulty_prior=raw.difficulty_prior, attrs=raw.attrs)

            b = mtor.inflate(beliefs[uid], ts)
            p = float(mtor.predict(b, [item])[0][0])
            sig = sim.answer(ui, raw.id, ts, evolve=True)
            if step >= warmup:
                P.append(p); P_o.append(sim.p_true(ui, raw.id)); Y.append(sig.outcome)
            beliefs[uid] = mtor.update(b, sig, item)
    return {
        "auc": auc(P, Y), "ece": expected_calibration_error(P, Y),
        "oracle_auc": auc(P_o, Y),
        "note": "single latent dimension, no taxonomy supplied",
    }


# --------------------------------------------------------------------------

def run_slope_recovery(n_items: int = 60, n_users: int = 300, steps: int = 80):
    """Does slope learning actually recover per-item discrimination?

    Run on a deliberately small catalogue so each item accrues real exposure.
    The main prequential run above averages ~20 exposures per item, which is far
    too thin: d p / d log(slope) is proportional to (theta - b), so an item
    answered by users near its own difficulty carries almost no slope signal.
    """
    sim = Simulator(SimConfig(n_items=n_items, n_users=n_users, seed=3))
    cfg = MTORConfig(floor_attr="floor", learn_discrimination=True, disc_min_exposure=10)
    mtor = MTOR(cfg, ItemStore(cfg))
    beliefs = {u: mtor.init(u, sim.space) for u in sim.users}

    for step in range(steps):
        ts = (step + 1) * DAY
        for ui, uid in enumerate(sim.users):
            item = sim.sample_items(1)[0]
            b = mtor.inflate(beliefs[uid], ts)
            beliefs[uid] = mtor.update(b, sim.answer(ui, item.id, ts, evolve=True), item)

    learned = np.array([mtor.items.disc(it) for it in sim.items])
    exposure = np.array([mtor.items.exposure(it) for it in sim.items])
    return {
        "exposure_mean": float(exposure.mean()),
        "corr_log_slope": float(np.corrcoef(np.log(learned), np.log(sim.item_disc))[0, 1]),
        "recovered_log_sd": float(np.log(learned).std()),
        "true_log_sd": float(np.log(sim.item_disc).std()),
        "note": "slope learning is an upside that switches on per item as a "
                "catalogue matures; it is never something a new tenant relies on",
    }


def _show(d, indent=2):

    pad = " " * indent
    for k, v in d.items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _show(v, indent + 2)
        elif isinstance(v, float):
            print(f"{pad}{k}: {v:.4f}")
        else:
            print(f"{pad}{k}: {v}")


def main():
    print("=" * 74)
    print("MTOR evaluation -- synthetic 3PL learners, estimator is Rasch (misspecified)")
    print("=" * 74)

    print("\n[1] Prequential online: calibration & discrimination (mean of 3 seeds)")
    keys = ("ece", "auc", "brier")
    models = ("mtor_no_floor", "mtor_floor", "mtor_floor_disc", "elo", "oracle")

    acc = {m: {k: [] for k in keys} for m in models}
    rel_no, rel_yes, n = None, None, 0
    for s in range(3):
        r = run_prequential(s)
        n = r["_n"]
        rel_no = rel_no or r["_rel_no_floor"]
        rel_yes = rel_yes or r["_rel_floor"]
        for m in models:
            for k in keys:
                acc[m][k].append(r[m][k])
    res = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    _show(res)
    print(f"  n_predictions_per_seed: {n}")
    o = res["oracle"]
    for m in ("mtor_no_floor", "mtor_floor", "mtor_floor_disc"):
        print(f"  -> {m:16s} vs oracle:  ECE gap {res[m]['ece'] - o['ece']:+.4f}   "
              f"AUC gap {o['auc'] - res[m]['auc']:+.4f}")
    print(f"  -> floor term:     ECE {res['mtor_no_floor']['ece']:.4f} -> "
          f"{res['mtor_floor']['ece']:.4f}   (oracle {o['ece']:.4f})")
    print(f"  -> slope term:     AUC {res['mtor_floor']['auc']:.4f} -> "
          f"{res['mtor_floor_disc']['auc']:.4f}   (oracle {o['auc']:.4f})")
    print(f"  -> best vs Elo:    ECE {res['mtor_floor_disc']['ece']:.4f} vs {res['elo']['ece']:.4f}"
          f"   AUC {res['mtor_floor_disc']['auc']:.4f} vs {res['elo']['auc']:.4f}")

    print("\n  reliability, low bins (the floor signature):  bin | n | p_hat | actual | diff")
    print("    -- no-floor variant --")
    for bucket, cnt, mp, ay in rel_no[:4]:
        print(f"      {bucket}  n={cnt:6d}  p_hat={mp:.3f}  actual={ay:.3f}  diff={ay - mp:+.3f}")
    print("    -- floor variant --")
    for bucket, cnt, mp, ay in rel_yes[:4]:
        print(f"      {bucket}  n={cnt:6d}  p_hat={mp:.3f}  actual={ay:.3f}  diff={ay - mp:+.3f}")


    print("\n[2] Ability recovery (ability frozen, 200 obs/user)")
    _show(run_ability_recovery())

    print("\n[3] CAT convergence -- fixed held-out probe set, RMSE vs oracle p")
    _show(run_cat_convergence())

    print("\n[4] Untagged degradation")
    _show(run_untagged())

    print("\n[5] Slope recovery under adequate exposure")
    _show(run_slope_recovery())


    print("\n" + "=" * 74)
    print("How to read: judge every number by its gap to the oracle, not by an")
    print("invented threshold. The oracle is capped well below perfect because a")
    print("guessing floor on multiple-choice items is irreducible noise.")
    print("=" * 74)


if __name__ == "__main__":
    main()
