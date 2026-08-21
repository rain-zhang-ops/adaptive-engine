"""Core contracts for the adaptive decision engine. Domain-agnostic.

Three irreducible primitives:

    Believe :  b_t = f(b_{t-1}, o_t)          posterior state given observations
    Score   :  u(a | b)                        utility of a candidate given state
    Choose  :  argmax_{A in F} U(A | b)        constrained subset selection

The entire configurable surface collapses into one utility functional:

    U(A | b) = E_b[ sum_{a in A} rho(o_a) + gamma * V(b') ] + Phi(A)

    rho    preference over outcomes        (peak / increasing / decreasing / threshold)
    gamma  exploit state vs. change state  (0 = immediate reward, >0 = pursue state gain)
    V      value of the successor belief   (mastery sum / negative entropy / info gain)
    Phi    set-structure functional        (diversify / concentrate / target entropy)

Scenario examples, all the same functional:

    adaptive testing   rho=constant(0), gamma=1, V=neg_entropy      pure information gain
    remediation        rho=peak(0.70),  gamma>0, V=mastery(low_mu),  Phi=diversify
    affinity boost     rho=increasing,  gamma=0,                     Phi=concentrate

IRON RULE
---------
No business vocabulary may appear in this module. If a term such as student,
question, knowledge point, homework, product or order shows up here, the
abstraction has leaked and the design has failed.

The only nouns are: user, item, tag, signal. State is called a profile/belief.

This module declares contracts only -- no algorithm lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Protocol, Sequence, runtime_checkable



import numpy as np

__all__ = [
    "TagSpace",
    "Item",
    "Signal",
    "Belief",
    "RewardSpec",
    "ValueSpec",
    "StructureSpec",
    "Utility",
    "Quota",
    "Constraints",
    "Scored",
    "Chosen",
    "Decision",
    "Believe",
    "Score",
    "Choose",
]


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagSpace:
    """Bijection between tag ids and vector indices.

    Dimensionality is never hard-coded; it grows as tags are registered.
    A customer may supply no tags at all, in which case the space holds the
    single reserved dimension ``_latent`` and every model degrades gracefully
    to a one-dimensional rating. Requiring a taxonomy up front would be an
    adoption blocker, so it must stay optional.
    """

    index_of: Mapping[str, int]
    tag_of: Sequence[str]
    parent_of: Mapping[str, str] = field(default_factory=dict)
    """Optional DAG edges. When present, Believe implementations MAY propagate
    updates along them; absence must never be an error."""

    LATENT: ClassVar[str] = "_latent"


    @property
    def n_dims(self) -> int:
        return len(self.tag_of)


@dataclass(frozen=True)
class Item:
    """A candidate action.

    Deliberately excludes content. The engine never receives item bodies,
    only metadata -- a privacy selling point and a compliance simplification
    under a strict-isolation deployment model.
    """

    id: str
    tag_weights: Mapping[str, float] = field(default_factory=dict)
    """Credit split across tags; should sum to 1.0. Empty means untagged, which
    routes the item to the reserved latent dimension."""

    difficulty_prior: float | None = None
    """Content-side prior used before the item accrues exposure. This is the
    only defence against the new-item deadlock (no exposure -> no estimate ->
    never selected -> no exposure)."""

    attrs: Mapping[str, Any] = field(default_factory=dict)
    """Opaque key-values. Constraints and quotas address these by path
    (e.g. ``attrs.kind``). The kernel never interprets their meaning."""


@dataclass(frozen=True)
class Signal:
    """One observed interaction."""

    user_id: str
    item_id: str
    outcome: float
    """Continuous in [0, 1]. Covers binary correctness, partial credit,
    completion ratio and normalised engagement with a single field."""

    ts: float
    context: Mapping[str, Any] = field(default_factory=dict)

    propensity: float | None = None
    """P(item selected | policy in force at emission time).

    Required for off-policy correction. Interactions are never a random
    sample -- they are whatever the previous policy chose -- so without this
    the system silently reinforces its own bias. Must be captured from day
    one; it cannot be reconstructed later.
    """


@dataclass
class Belief:
    """Posterior over a user's latent state, per tag dimension."""

    user_id: str
    mu: np.ndarray
    var: np.ndarray
    """Uncertainty is mandatory, not decoration. Active selection, Thompson
    exploration, cold-start shrinkage and calibrated confidence all read from
    it; a point-estimate model (plain Elo) cannot support them."""

    last_seen: np.ndarray
    """Per-dimension timestamp driving uncertainty inflation."""

    space: TagSpace
    model_version: str


# ---------------------------------------------------------------------------
# Utility  ==  (rho, gamma, V, Phi)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardSpec:
    """rho -- preference over the predicted outcome of a single item."""

    kind: Literal["peak", "increasing", "decreasing", "threshold", "constant"]
    target: float | None = None
    width: float = 0.15
    value_attr: str | None = None
    """Optional item attribute path multiplying the reward (unit price,
    score weight, ...). Keeps monetary/weighting concerns out of the kernel."""


@dataclass(frozen=True)
class ValueSpec:
    """V -- value assigned to the successor belief. Only consulted when gamma > 0."""

    kind: Literal["mastery_sum", "neg_entropy", "info_gain"]
    dim_weight: Literal["uniform", "low_mu", "high_mu", "high_var", "target"] = "uniform"
    target: Mapping[str, float] | None = None
    """Required when dim_weight == "target": the desired profile to shape toward."""


@dataclass(frozen=True)
class StructureSpec:
    """Phi -- functional over the chosen set as a whole.

    Diversity is not unconditionally good: remediation wants coverage, while
    affinity amplification wants concentration. Hence the two are peer
    strategies rather than one signed weight.
    """

    kind: Literal["diversify", "concentrate", "balanced"]
    weight: float = 0.0
    similarity: Literal["tag_jaccard", "tag_cosine"] = "tag_jaccard"
    target_entropy: float | None = None


@dataclass(frozen=True)
class Utility:
    rho: RewardSpec
    gamma: float = 0.0
    value: ValueSpec | None = None
    structure: StructureSpec | None = None

    explore_floor: float = 0.0
    """Minimum share of the result set reserved for high-variance items.

    Serves two purposes with one mechanism: correcting selection bias in the
    training data, and preventing pure amplification from collapsing into a
    filter bubble. Policies that only exploit MUST NOT set this to zero.
    """

    transition_model: str | None = None
    """Plugin id estimating how the belief moves after acting. Domain-specific
    by nature (skill acquisition differs from preference drift), so it is an
    explicit extension point rather than a pretence of generality. Must be
    supplied whenever gamma > 0.
    """


# ---------------------------------------------------------------------------
# Feasible set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quota:
    """Per-group cardinality requirement, e.g. group_by="attrs.kind"."""

    group_by: str
    counts: Mapping[str, int]


@dataclass(frozen=True)
class Constraints:
    """F -- the feasible set. Hard filters, never soft penalties.

    Anything that must not be returned under any score (visibility class,
    tenant boundary, embargoed items) belongs here and is applied during
    recall. Encoding such rules as score penalties is how leaks happen.
    """

    k: int
    predicates: Sequence[str] = ()
    """Boolean expressions over ``item.*`` / ``attrs.*``, evaluated in a
    sandbox. The kernel does not know what they mean."""

    quotas: Sequence[Quota] = ()
    max_per_tag: int | None = None
    exclude_item_ids: frozenset[str] = frozenset()
    within_tags: frozenset[str] = frozenset()
    """Restrict the feasible set to items carrying at least one of these tags.
    Empty means no restriction. A hard filter for the same reason the rest of
    this class is one: a caller who scopes a request to a tag set and silently
    receives items from outside it has no way to notice."""



# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    item_id: str
    p_hat: float
    sigma: float
    utility: float


@dataclass(frozen=True)
class Chosen:
    item_id: str
    utility: float
    p_hat: float
    marginal_gain: float

    propensity: float
    """Probability this item was selected under the acting policy. Echoed back
    on the wire so the caller can return it with future signals."""

    reasons: Mapping[str, float]
    """Machine-readable contributions. Keys produced by the chooser:

    * ``fit``        -- rho contribution (how well the predicted outcome matches
                        what the goal wants).
    * ``structure``  -- the set functional Phi's marginal contribution; negative
                        under ``diversify`` (a redundant item is penalised).
    * ``sigma``      -- posterior standard deviation, the raw uncertainty behind
                        an explore pick.
    * ``explore``    -- whether this slot came from the exploration reserve.
    * ``propensity_exact`` -- 1.0 if this item's ``propensity`` is analytic, 0.0
                        if it is an approximation (a quota/tag cap perturbed the
                        exploration draw). IPS consumers must read this before
                        trusting the weight.

    Rendering these into human sentences is the API layer's job, not the
    kernel's -- the kernel has no vocabulary to render into."""



@dataclass(frozen=True)
class Decision:
    chosen: Sequence[Chosen]
    confidence: Literal["high", "medium", "low"]
    fallback_reason: str | None = None
    """Set when the engine degraded (unknown user, empty candidates,
    insufficient data). Degradation is always preferred over raising: a 5xx
    from the engine is an outage in the caller's product."""

    model_version: str = ""
    policy_id: str = ""


# ---------------------------------------------------------------------------
# The three primitives
# ---------------------------------------------------------------------------


@runtime_checkable
class Believe(Protocol):
    """Observation -> posterior state. The pluggable seam.

    Implementations range from training-free online rating (usable on day
    one, which matters when no cross-customer pretraining is permitted) to
    sequence models that switch on past a data threshold.
    """

    version: str

    def init(self, user_id: str, space: TagSpace, prior: np.ndarray | None = None) -> Belief:
        ...

    def update(self, belief: Belief, signal: Signal, item: Item) -> Belief:
        """Fold one observation in.

        Implementations MUST distribute credit across ``item.tag_weights``
        rather than picking a single dimension: a multi-tag item that went
        wrong has no single culprit, and attributing in proportion to
        uncertainty is what makes multi-tag data usable at all.
        """

    def inflate(self, belief: Belief, now: float) -> Belief:
        """Apply elapsed-time effects.

        Inflate variance rather than decaying the mean. Time since last
        observation means "we are less sure whether this still holds", not
        "this has certainly been lost". Mean decay, where a domain calls for
        it, belongs in an optional plugin.
        """


@runtime_checkable
class Score(Protocol):
    """State + candidates -> utility."""

    def predict(self, belief: Belief, items: Sequence[Item]) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_hat, sigma).

        Predictions MUST shrink toward 0.5 as variance grows, so that a
        thinly-observed user never receives over-confident estimates. This is
        what makes a calibration guarantee expressible as a contractual SLA.
        """

    def value(
        self,
        belief: Belief,
        items: Sequence[Item],
        utility: Utility,
    ) -> Sequence[Scored]:
        """Per-item utility from rho and gamma*V. Set-level Phi is Choose's job."""


@runtime_checkable
class Choose(Protocol):
    """Utility + feasible set -> action set."""

    def solve(
        self,
        scored: Sequence[Scored],
        items: Sequence[Item],
        utility: Utility,
        constraints: Constraints,
    ) -> Decision:
        """Select k items maximising U(A | b) subject to F.

        Exact optimisation over a combinatorial action space under partial
        observability is intractable. The sanctioned approximation is
        one-step lookahead plus greedy selection, which carries a (1 - 1/e)
        guarantee while Phi stays submodular. That bound is part of the
        contract and should be documented rather than hidden.
        """
