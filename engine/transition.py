"""Transition models -- how the belief is expected to move after acting.

``gamma > 0`` means the objective is to *change* the state rather than exploit
it, and that requires an estimate of how acting changes it. Two parts, with very
different epistemic status, and conflating them is the main trap:

**Variance is exact.** Under the same ADF update MTOR already performs, the
posterior variance after observing an item is known in closed form:

    var' = 1 / (1/var + fisher),      fisher = (a * slope)^2 / (p(1-p))

No domain assumption enters. Information gain, adaptive testing and
uncertainty-driven exploration all rest on this and are therefore safe.

**Mean movement is a domain hypothesis.** Under a pure Bayesian filter the
posterior mean is a martingale: E[mu'] = mu, so the *expected* mastery gain from
merely observing is exactly zero. Any positive expected gain comes from the
claim that acting changes the underlying state -- practice teaches, exposure
shifts preference -- and that claim is domain knowledge, not mathematics.

So each model here declares its own hypothesis about the *shape* f(p) of that
gain, and the shape is the whole content:

    E[Delta mu_k] = eta * a_k * f(p)

``eta`` (how much one interaction is worth) is a single positive constant shared
by every candidate, so it cancels in any comparison and merges into ``gamma``.
That is worth stating plainly: **eta needs no calibration for ranking**, only
f(p) does. What must be calibrated is which f(p) is true for the domain, and
``engine/calibrate.py`` is the tool that answers it -- on the reference
simulator it found f monotone increasing, and explicitly refused to endorse the
peaked shape.

There is deliberately no model named ``default``. A name like that hides which
domain hypothesis a deployment signed up for; the config has to say it out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

__all__ = ["TransitionModel", "get_transition", "TRANSITIONS", "TransitionError"]


class TransitionError(KeyError):
    pass


@dataclass(frozen=True)
class TransitionModel:
    id: str
    shape: Callable[[np.ndarray], np.ndarray]
    """f(p) -- expected mean movement per unit tag weight, up to the shared
    constant eta. Must be non-negative and bounded by 1 so that V stays on the
    same [0,1] scale as rho and gamma keeps a transferable meaning."""

    hypothesis: str
    status: str
    """``exact`` = no domain assumption; ``hypothesis`` = a claim about the world
    that calibration has not confirmed; ``supported`` = confirmed on data."""

    @staticmethod
    def variance_after(var: np.ndarray, fisher: np.ndarray) -> np.ndarray:
        """Exact ADF posterior variance. Shared by every model -- the part that
        is mathematics, not assumption."""
        return 1.0 / (1.0 / np.maximum(var, 1e-12) + np.maximum(fisher, 0.0))

    def mean_shift(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.shape(np.asarray(p, dtype=float)), 0.0, 1.0)


def _martingale(p: np.ndarray) -> np.ndarray:
    # A Bayesian filter alone moves the mean nowhere in expectation. Correct,
    # and it means gamma*V is pure information under this model.
    return np.zeros_like(p)


def _monotone_gain(p: np.ndarray) -> np.ndarray:
    # "Succeeding consolidates, failing costs" -- gain rises with success odds.
    # This is the reference simulator's own learn rule (0.4 + 0.6*outcome, whose
    # expectation is 0.4 + 0.6p) and it is the shape calibrate.py actually found.
    return 0.4 + 0.6 * p


def _desirable_difficulty(p: np.ndarray) -> np.ndarray:
    # The textbook claim that the most learning happens where the outcome is
    # genuinely uncertain. Normalised to peak at 1.0 when p = 0.5.
    # Kept available but flagged: on the reference simulator, calibration found
    # NO interior peak, so selecting this is asserting something unverified.
    return 4.0 * p * (1.0 - p)


def _uncertainty_only(p: np.ndarray) -> np.ndarray:
    # Movement in *either* direction: E|r - p| = 2p(1-p). Not a gain claim --
    # useful when the goal is to resolve a state, not improve it.
    return 2.0 * p * (1.0 - p)


TRANSITIONS: Mapping[str, TransitionModel] = {
    "info_only": TransitionModel(
        id="info_only", shape=_martingale, status="exact",
        hypothesis="Observing alone does not move the mean (martingale). "
                   "Value comes entirely from variance reduction."),
    "monotone_gain": TransitionModel(
        id="monotone_gain", shape=_monotone_gain, status="supported",
        hypothesis="Expected state gain increases with success probability. "
                   "Confirmed on the reference simulator (engine/calibrate.py: "
                   "gain curve monotone over p in [0.30, 0.90])."),
    "desirable_difficulty": TransitionModel(
        id="desirable_difficulty", shape=_desirable_difficulty, status="hypothesis",
        hypothesis="Most state gain occurs where the outcome is most uncertain. "
                   "NOT confirmed: calibration found no interior peak. Selecting "
                   "this asserts a domain effect the data has not shown."),
    "state_resolution": TransitionModel(
        id="state_resolution", shape=_uncertainty_only, status="exact",
        hypothesis="Expected absolute mean movement E|r-p| = 2p(1-p). A claim "
                   "about how much the estimate will move, not about improvement."),
}


def get_transition(name: str | None) -> TransitionModel:
    if not name:
        raise TransitionError(
            "gamma > 0 requires an explicit transition_model; there is no default "
            "because the choice is a domain claim")
    if name not in TRANSITIONS:
        raise TransitionError(
            f"unknown transition_model {name!r}; available: {sorted(TRANSITIONS)}")
    return TRANSITIONS[name]
