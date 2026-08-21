"""Score.value -- turns a belief plus candidates into per-item utility.

Implements the first two terms of

    U(A | b) = E_b[ sum_a rho(o_a) + gamma * V(b') ] + Phi(A)

Phi is a property of the chosen *set*, so it lives in the chooser. Everything
here is per-item and therefore fully vectorisable.

Both terms are normalised to [0, 1] on purpose. Without that, ``gamma`` would
not mean anything transferable: a reward on one scale and a value on another
makes the same gamma behave differently per deployment, which is exactly how
configuration turns into folklore.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from contracts.core import Belief, Item, RewardSpec, Scored, Utility, ValueSpec
from engine.mtor import MTOR, TagLayout
from engine.transition import TransitionModel, get_transition


__all__ = ["UtilityScorer"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _get_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return None
    return cur


class UtilityScorer:
    """Implements ``contracts.core.Score``; delegates prediction to a Believe."""

    def __init__(self, believe: MTOR) -> None:
        self.believe = believe

    # -- Score.predict ----------------------------------------------------

    def predict(self, belief: Belief, items: Sequence[Item]) -> tuple[np.ndarray, np.ndarray]:
        return self.believe.predict(belief, items)

    # -- rho --------------------------------------------------------------

    def reward(self, p: np.ndarray, items: Sequence[Item], spec: RewardSpec) -> np.ndarray:
        kind = spec.kind
        if kind == "peak":
            target = 0.7 if spec.target is None else spec.target
            w = max(spec.width, 1e-6)
            r = np.exp(-((p - target) ** 2) / (2.0 * w * w))
        elif kind == "increasing":
            r = p.copy()
        elif kind == "decreasing":
            r = 1.0 - p
        elif kind == "threshold":
            t = 0.5 if spec.target is None else max(spec.target, 1e-6)
            r = np.minimum(1.0, p / t)
        elif kind == "constant":
            # Adaptive testing: the outcome itself is irrelevant, only what it
            # reveals. All information then has to come from gamma * V.
            r = np.zeros_like(p)
        else:
            raise ValueError(f"unknown reward kind {kind!r}")

        if spec.value_attr:
            # An item missing the attribute, or carrying a non-finite / negative
            # value, would otherwise silently zero or invert its reward -- a filter
            # nobody asked for, or an rho pushed outside [0, 1] that corrupts the
            # greedy ordering. Clamp to >= 0 and treat missing as "no boost" (1.0),
            # not "excluded" (0.0).
            raw = [_get_path(it, spec.value_attr) for it in items]
            vals = np.array([float(v) if isinstance(v, (int, float)) and math.isfinite(v)
                             else float("nan") for v in raw], dtype=float)
            missing = np.isnan(vals)
            vals = np.where(missing | (vals < 0.0), np.where(missing, np.nan, 0.0), vals)
            finite = vals[~np.isnan(vals)]
            hi = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
            scale = np.where(np.isnan(vals), 1.0, vals / hi)
            r = r * scale          # stays within [0, 1]
        return r


    # -- gamma * V --------------------------------------------------------

    def dim_weights(self, belief: Belief, spec: ValueSpec) -> np.ndarray:
        mode = spec.dim_weight
        level = _sigmoid(belief.mu)                     # logits -> [0,1]
        if mode == "uniform":
            return np.ones_like(level)
        if mode == "low_mu":
            return 1.0 - level                          # remediation
        if mode == "high_mu":
            return level                                # amplification
        if mode == "high_var":
            hi = float(belief.var.max())
            return belief.var / hi if hi > 0 else np.ones_like(level)
        if mode == "target":
            if not spec.target:
                raise ValueError("dim_weight='target' requires ValueSpec.target")
            tgt = np.zeros_like(level)
            for tag, v in spec.target.items():
                j = belief.space.index_of.get(tag)
                if j is not None:
                    tgt[j] = v
            return np.abs(tgt - level)
        raise ValueError(f"unknown dim_weight {mode!r}")

    def recall_weights(self, belief: Belief, utility: Utility) -> np.ndarray:
        """Per-dimension weights for *candidate generation*, defined for every
        utility -- including the ones with no ValueSpec at all.

        Recall has to know which dimensions matter before scoring happens, and
        ``gamma == 0`` policies (pure exploitation) carry no ValueSpec to read it
        from. Rather than defaulting those to uniform -- which makes recall
        random and the objective irrelevant -- the direction is derived from rho:

            increasing  -> high_mu    we want likely success, so strong dimensions
            decreasing  -> low_mu     we want likely failure, so weak dimensions
            peak / constant -> high_var   items near a mid success rate, and the
                                          most informative ones, live where the
                                          estimate is least settled

        Stated here rather than inside the recall query so the mapping is visible
        next to the rest of the weighting logic.
        """
        if utility.value is not None:
            return self.dim_weights(belief, utility.value)
        kind = utility.rho.kind
        mode = {"increasing": "high_mu", "decreasing": "low_mu"}.get(kind, "high_var")
        return self.dim_weights(belief, ValueSpec(kind="mastery_sum", dim_weight=mode))

    def value(

        self,
        belief: Belief,
        items: Sequence[Item],
        utility: Utility,
    ) -> list[Scored]:
        # One tag-incidence pass for the whole pool, shared with _value_term
        # below. Both used to resolve every item's tags independently, which was
        # the largest single cost of a decide once the arithmetic was vectorised.
        layout = self.believe.tag_layout(items, belief.space)
        p, sigma = self.believe.predict(belief, items, layout)
        rho = self.reward(p, items, utility.rho)

        if utility.gamma <= 0.0 or utility.value is None:
            total = rho
        else:
            trans = get_transition(utility.transition_model)
            total = rho + utility.gamma * self._value_term(
                belief, items, p, utility.value, trans, layout)

        return [
            Scored(item_id=it.id, p_hat=float(p[i]), sigma=float(sigma[i]), utility=float(total[i]))
            for i, it in enumerate(items)
        ]

    def components(
        self, belief: Belief, items: Sequence[Item], utility: Utility
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(p_hat, sigma, rho, gamma*V) -- kept separate so the chooser can put
        an honest per-term breakdown into its explanation."""
        layout = self.believe.tag_layout(items, belief.space)
        p, sigma = self.believe.predict(belief, items, layout)
        rho = self.reward(p, items, utility.rho)
        if utility.gamma <= 0.0 or utility.value is None:
            return p, sigma, rho, np.zeros_like(rho)
        trans = get_transition(utility.transition_model)
        return (p, sigma, rho,
                utility.gamma * self._value_term(
                    belief, items, p, utility.value, trans, layout))

    def _value_term(
        self, belief: Belief, items: Sequence[Item], p: np.ndarray, spec: ValueSpec,
        trans: TransitionModel, layout: TagLayout | None = None,
    ) -> np.ndarray:
        lay = layout if layout is not None else self.believe.tag_layout(items, belief.space)
        w = self.dim_weights(belief, spec)
        out = np.zeros(lay.n, dtype=float)
        m = lay.m
        if m == 0:
            return out
        shift = trans.mean_shift(p)          # f(p), the domain hypothesis

        if spec.kind == "mastery_sum":
            # Expected state gain = eta * sum_k a_k * f(p), with the shared
            # constant eta folded into gamma. The shape f(p) comes from the
            # declared transition model rather than being written in here, so the
            # domain claim stays visible in configuration.
            #
            # sum_k a_k * w_k is a bincount over the flat (item, tag) pairs --
            # the same arithmetic the per-item dot product did.
            gain = np.bincount(lay.rows, weights=lay.vals * w[lay.cols], minlength=m)
            out[lay.tagged] = gain * shift[lay.tagged]
        elif spec.kind in ("neg_entropy", "info_gain"):
            # Fractional posterior variance reduction, in [0, 1). Exact under the
            # same ADF update MTOR performs -- no domain assumption, which is why
            # adaptive testing needs no separate strategy. Still per item: the
            # reduction is a ratio inside each item's own dimensions, so it does
            # not reduce to one flat bincount. The tag lookup is reused, though.
            for k, i in enumerate(lay.tagged.tolist()):
                sl = lay.pair_slice(k)
                idx, a = lay.cols[sl], lay.vals[sl]
                _, gz, c, s = self.believe._predict_one(belief, items[i], idx, a)
                slope = (1.0 - c) * s * (1.0 - s) * gz
                denom = max(p[i] * (1.0 - p[i]), 1e-9)
                fisher = (a ** 2) * (slope ** 2) / denom
                var = belief.var[idx]
                reduction = 1.0 - TransitionModel.variance_after(var, fisher) / np.maximum(var, 1e-12)
                out[i] = float(np.dot(w[idx] / max(w[idx].sum(), 1e-9), reduction))
        else:
            raise ValueError(f"unknown value kind {spec.kind!r}")
        return np.clip(out, 0.0, 1.0)

