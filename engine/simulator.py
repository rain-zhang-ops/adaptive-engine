"""Synthetic learner simulator.

Two jobs:

1. Validate the engine with zero real data. Calibration, credit assignment and
   convergence can all be measured against known ground truth, which is
   impossible with observational data alone.
2. Under a strict-isolation deployment (no cross-customer pretraining),
   synthetic data is the only lawful route to a prior. Nothing here belongs to
   any customer.

Deliberate misspecification
---------------------------
The simulator is a 3PL model -- per-item discrimination and a guessing floor --
while MTOR is Rasch-style with neither. This is on purpose. If the generator
and the estimator shared a functional form, calibration error would be
trivially near zero and the measurement would be worthless. The gap here is
the same kind of gap that exists between any model and reality.

    p_true = c_i + (1 - c_i) * sigmoid(disc_i * (theta - b_i))

Ground truth ability also moves: practice raises it with diminishing returns,
and idleness pulls it back toward a per-tag baseline at a per-tag rate. A
single global forgetting constant cannot represent that, which is exactly why
MTOR inflates variance instead of hard-coding a decay curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from contracts.core import Item, Signal, TagSpace

__all__ = ["SimConfig", "Simulator"]

_SECONDS_PER_DAY = 86400.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass(frozen=True)
class SimConfig:
    n_tags: int = 24
    n_items: int = 600
    n_users: int = 200

    max_tags_per_item: int = 3
    """Multi-tag items are the norm in practice, and they are what breaks naive
    single-skill models. Keep them in the generator."""

    guess_kinds: tuple[str, ...] = ("choice", "blank", "essay")
    guess_floor: dict = field(default_factory=lambda: {"choice": 0.25, "blank": 0.05, "essay": 0.0})
    """A 4-option multiple choice item can be answered correctly by luck. MTOR
    has no guessing parameter, so this is a genuine source of misspecification."""

    disc_log_sigma: float = 0.35
    """Item discrimination ~ lognormal(0, sigma). MTOR assumes all items
    discriminate equally, so this is misspecification too."""

    ability_sigma: float = 1.0
    difficulty_sigma: float = 1.0

    learn_rate: float = 0.06
    """Practice gain per interaction, damped as ability rises."""

    forget_rate_lo: float = 0.002
    forget_rate_hi: float = 0.02
    """Per-tag pull-back per idle day, sampled in this range. Heterogeneous on
    purpose."""

    seed: int = 20260820


class Simulator:
    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        c = self.cfg

        self.tags = [f"t{k:03d}" for k in range(c.n_tags)]
        self.space = TagSpace(
            index_of={t: i for i, t in enumerate(self.tags)},
            tag_of=list(self.tags),
        )

        # -- items ---------------------------------------------------------
        self.item_b = self.rng.normal(0.0, c.difficulty_sigma, c.n_items)
        self.item_disc = np.exp(self.rng.normal(0.0, c.disc_log_sigma, c.n_items))
        kinds = self.rng.choice(c.guess_kinds, size=c.n_items, p=[0.6, 0.3, 0.1])
        self.item_kind = kinds
        self.item_guess = np.array([c.guess_floor[k] for k in kinds], dtype=np.float64)

        self.item_tags: list[dict[str, float]] = []
        for _ in range(c.n_items):
            k = int(self.rng.integers(1, c.max_tags_per_item + 1))
            picks = self.rng.choice(c.n_tags, size=k, replace=False)
            w = self.rng.dirichlet(np.full(k, 2.0))
            self.item_tags.append({self.tags[int(p)]: float(x) for p, x in zip(picks, w)})

        self.items: list[Item] = [
            Item(
                id=f"i{j:05d}",
                tag_weights=self.item_tags[j],
                # A noisy content-side hint, as a human/LLM annotation would be.
                difficulty_prior=float(np.clip(
                    0.5 + (self.item_b[j] + self.rng.normal(0.0, 0.4)) / 4.0, 0.0, 1.0)),
                # Also a noisy annotation, not the exact generating floor. Handing
                # MTOR the true guess floor would make its `c` an oracle and the
                # calibration numbers optimistic in a way a real adapter (which
                # guesses "4-choice => 0.25") can never reproduce. The floor is
                # heterogeneous like difficulty is, so it must be estimated from an
                # approximate hint, not shared.
                attrs={"kind": str(kinds[j]),
                       "floor": float(np.clip(
                           c.guess_floor[kinds[j]] + self.rng.normal(0.0, 0.05),
                           0.0, 0.95))},

            )
            for j in range(c.n_items)
        ]

        self.item_index = {it.id: j for j, it in enumerate(self.items)}

        # -- users ---------------------------------------------------------
        self.ability = self.rng.normal(0.0, c.ability_sigma, (c.n_users, c.n_tags))
        self.baseline = self.ability.copy()
        self.forget = self.rng.uniform(c.forget_rate_lo, c.forget_rate_hi, c.n_tags)
        self.last_practice = np.zeros((c.n_users, c.n_tags), dtype=np.float64)
        self.users = [f"u{i:05d}" for i in range(c.n_users)]

    # -- ground truth ------------------------------------------------------

    def theta(self, user: int, item_id: str) -> float:
        j = self.item_index[item_id]
        w = self.item_tags[j]
        return float(sum(x * self.ability[user, self.space.index_of[t]] for t, x in w.items()))

    def p_true(self, user: int, item_id: str) -> float:
        j = self.item_index[item_id]
        core = _sigmoid(self.item_disc[j] * (self.theta(user, item_id) - self.item_b[j]))
        return float(self.item_guess[j] + (1.0 - self.item_guess[j]) * core)

    # -- interaction -------------------------------------------------------

    def answer(self, user: int, item_id: str, ts: float, *, evolve: bool = True) -> Signal:
        """Draw one observation. ``evolve=False`` freezes ability, which is what
        the convergence test needs (a moving target cannot be converged to)."""
        if evolve:
            self._forget(user, ts)
        p = self.p_true(user, item_id)
        outcome = 1.0 if self.rng.random() < p else 0.0
        if evolve:
            self._learn(user, item_id, ts, outcome)
        return Signal(user_id=self.users[user], item_id=item_id, outcome=outcome, ts=ts)

    def _learn(self, user: int, item_id: str, ts: float, outcome: float) -> None:
        j = self.item_index[item_id]
        for t, w in self.item_tags[j].items():
            k = self.space.index_of[t]
            headroom = 1.0 / (1.0 + np.exp(self.ability[user, k] - 2.0))
            gain = self.cfg.learn_rate * w * headroom * (0.4 + 0.6 * outcome)
            self.ability[user, k] += gain
            self.baseline[user, k] += 0.5 * gain
            self.last_practice[user, k] = ts

    def _forget(self, user: int, ts: float) -> None:
        seen = self.last_practice[user]
        days = np.where(seen > 0.0, (ts - seen) / _SECONDS_PER_DAY, 0.0)
        np.maximum(days, 0.0, out=days)
        pull = np.minimum(self.forget * days, 1.0)
        self.ability[user] += pull * (self.baseline[user] - self.ability[user])

    # -- helpers -----------------------------------------------------------

    def item_by_id(self, item_id: str) -> Item:
        return self.items[self.item_index[item_id]]

    def sample_items(self, n: int) -> list[Item]:
        picks = self.rng.choice(len(self.items), size=min(n, len(self.items)), replace=False)
        return [self.items[int(p)] for p in picks]

    def true_ability(self, user: int) -> np.ndarray:
        return self.ability[user].copy()

    def untagged_items(self, items: Sequence[Item]) -> list[Item]:
        """Strip tags, to exercise the no-taxonomy degradation path."""
        return [Item(id=it.id, tag_weights={}, difficulty_prior=it.difficulty_prior,
                     attrs=it.attrs) for it in items]
