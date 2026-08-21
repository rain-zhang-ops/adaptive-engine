"""MTOR -- Multi-Trait Online Rating.

A training-free ``Believe`` implementation: assumed density filtering (ADF) on a
Rasch-style logistic observation model, with per-dimension Gaussian posteriors.

Why this and not plain Elo
--------------------------
Elo carries a point estimate only. Every capability the product depends on --
active selection, Thompson exploration, cold-start shrinkage, calibrated
confidence -- reads from the variance. Without it the engine degrades into a
ranker.

Observation model
-----------------
    theta_si = sum_k a_ik * mu_sk                       ability aggregated over item tags
    var_si   = sum_k a_ik^2 * var_sk                    (independence assumption)
    g        = 1 / sqrt(1 + LAMBDA * (var_si + var_bi))  uncertainty attenuation
    p_hat    = sigmoid(g * (theta_si - b_i))

``g`` pulls predictions toward 0.5 as uncertainty grows, so a thinly observed
user never gets an over-confident estimate. That property is what makes a
calibration guarantee expressible as a contractual SLA.

Update (ADF / Laplace, observing r in [0, 1])
---------------------------------------------
    mu_sk   += var_sk * a_ik * g * (r - p_hat)
    var_sk   = 1 / (1/var_sk + (a_ik * g)^2 * p_hat * (1 - p_hat))

Note the mean step is *already* proportional to ``var_sk``: credit for a
multi-tag observation lands on the dimensions we are least sure about. That is
not a hand-designed heuristic -- it falls out of the Bayesian update. A
multi-tag item answered wrong has no single culprit, and this is the principled
way to apportion blame.

Elapsed time inflates variance instead of decaying the mean:

    var_sk += drift_k^2 * delta_days

"time since last observation" means "we are less sure whether this still
holds", not "this has certainly been lost". Mean decay, where a domain calls
for it, belongs in a separate optional plugin.

Every constant is either derived (see LOGIT_VAR_CORRECTION) or flagged
uncalibrated in MTORConfig. No arbitrary magic numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from contracts.core import Belief, Item, Scored, Signal, TagSpace

__all__ = ["MTORConfig", "ItemStore", "MTOR", "TagLayout"]


# The variance attenuation factor for a logistic link under a Gaussian latent.
# Matching the logistic and probit slopes gives 3/pi^2; this is the same
# correction Glicko applies. Derived, not tuned.
LOGIT_VAR_CORRECTION = 3.0 / (math.pi ** 2)

_SECONDS_PER_DAY = 86400.0


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _clip1(x: float, lo: float, hi: float) -> float:
    """``float(np.clip(x, lo, hi))`` without numpy's dispatch, for scalars.

    Identical for every input including NaN and the infinities, which is why the
    NaN test is here rather than a bare ``min``/``max``: those propagate NaN only
    depending on argument order, and a NaN difficulty silently poisons every
    prediction the item takes part in. The reason to avoid ``np.clip`` is cost --
    it is ~5us on one Python float against ~0.1us here, and it runs once per
    candidate per request, so it was 18% of a decide.
    """
    x = float(x)
    if x != x:
        return x
    return lo if x < lo else hi if x > hi else x



class TagLayout(NamedTuple):
    """A candidate pool's (item, tag) incidence, flattened once.

    Resolving an item's tags into (dimension indices, normalised weights) is pure
    Python dict work, and it used to happen twice per item per decision -- once in
    ``MTOR.predict`` and again in ``UtilityScorer._value_term``. At a 2000-item
    pool that was the single largest remaining cost of a decide (~21%) and none of
    it was arithmetic. Building the layout once and handing it to both callers
    removes the duplicate pass outright.

    Layout is *not* an (items x dims) matrix on purpose: that is O(pool x
    taxonomy) memory. These arrays are O(tags actually present), which is all the
    reductions need.

    ``tagged`` holds pool positions that carry at least one known tag, ascending.
    Untagged items are absent from every array here; callers give them the prior.
    ``rows`` indexes into ``tagged`` (i.e. it is compacted), so a ``bincount`` over
    it yields one entry per tagged item. ``starts``/``counts`` slice the flat
    arrays back into per-item views for the few paths that still need them.
    """

    n: int
    tagged: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    starts: np.ndarray
    counts: np.ndarray

    @property
    def m(self) -> int:
        """Number of tagged items -- the length of every reduction below."""
        return int(self.tagged.size)

    def pair_slice(self, k: int) -> slice:
        """The (cols, vals) span belonging to tagged item ``k``."""
        s = int(self.starts[k])
        return slice(s, s + int(self.counts[k]))



@dataclass(frozen=True)
class MTORConfig:
    """All tunables in one place.

    provenance: UNCALIBRATED unless noted. These are structural defaults, not
    fitted values; they must be calibrated against real interaction data before
    any number derived from them is published or written into an SLA.
    """

    prior_mu: float = 0.0
    prior_var: float = 1.0
    """Latent scale is fixed by the prior: ability ~ N(0, 1) in logit units."""

    item_prior_var: float = 0.5
    """Items start better-known than users -- an item's difficulty is a property
    of the item, observed across all users, so it converges much faster."""

    difficulty_spread: float = 2.0
    """Maps a content-side difficulty_prior in [0,1] onto b in
    [-spread, +spread]. This is the only defence against the new-item deadlock
    (no exposure -> no estimate -> never selected -> no exposure)."""

    drift_per_day: float = 0.01
    """Variance growth per elapsed day. Per-tag overrides live in
    ``drift_overrides``: different material is forgotten at different rates,
    which is precisely what a single global forgetting constant cannot express."""

    drift_overrides: Mapping[str, float] = field(default_factory=dict)

    max_var: float = 1.5
    """Cap on inflation. Uncertainty must not exceed 'we know nothing' by much,
    otherwise long-dormant users produce numerically unstable updates."""

    learn_item_difficulty: bool = True

    learn_discrimination: bool = False
    """Learn a per-item slope (how sharply success separates competent from
    incompetent users).

    Off by default because it is the least stable parameter in the model: it is
    multiplicative, so early noisy estimates distort every prediction that item
    touches. Guarded three ways -- a tight prior, a minimum exposure count
    before learning starts, and a hard clip on the log-slope.

    Turn it on once items accumulate exposure. The payoff is the discrimination
    half of the AUC gap; the risk is instability on thin data, hence the gate.
    """

    disc_prior_var: float = 0.30
    """Calibrated, not guessed. On synthetic data with adequate exposure the
    recovered spread of log-slope is sd 0.30 at this setting versus 0.14 at
    0.05, against a true spread of 0.38 -- so 0.05 over-shrinks. Correlation
    with the true slope is ~0.66 either way, meaning the tighter prior costs
    scale without buying accuracy."""

    disc_min_exposure: int = 200
    """No slope learning until an item has been answered this many times.

    Set from measurement, not taste. At ~20 exposures per item, enabling slope
    learning moves AUC by 0.0000 -- the gradient d p / d log(slope) is
    proportional to (theta - b), so items answered by users near their own
    difficulty carry almost no slope signal, and it takes a large, ability-spread
    sample to accumulate any. At ~400 exposures the recovered slope correlates
    +0.67 with truth. This matches the standard result that a two-parameter
    item model needs an order of magnitude more responses per item than a
    one-parameter one.

    Consequence for the product: slope learning is an upside that switches on
    per item as the catalogue matures, never something a new tenant relies on.
    """


    disc_log_clip: float = 0.7
    """Clip log-slope to +/- this, i.e. slope in roughly [0.5, 2.0]. A slope
    near zero would make an item carry no information yet still absorb updates;
    a huge slope turns one observation into a near-deterministic claim."""

    floor_attr: str | None = None

    """Name of an item attribute holding a *baseline success probability* -- the
    chance of success with zero competence.

    Without this term the model asserts that success probability can approach 0
    for a hard item, which is false whenever partial credit by luck exists. The
    signature of omitting it is unmistakable in a reliability table: predictions
    in the low bins come out systematically under actual rates while high bins
    stay unbiased.

    The kernel does not interpret why a floor exists -- it reads a number from
    an opaque attribute. Deciding that a 4-option item has floor 0.25, or that a
    channel converts at 3% regardless of affinity, is the adapter's job. Leave
    as None and the model reduces exactly to the plain Rasch form.
    """

    min_var: float = 1e-4



class ItemStore:
    """Mutable item-side parameters: difficulty, slope, and their uncertainties.

    Kept out of ``contracts.core.Item`` on purpose -- Item is a frozen
    description supplied by the caller, while these are learned state.

    The slope is held in log space so it stays positive without constrained
    optimisation; a negative slope would mean "more competence lowers success
    probability", which is never what we intend to fit.
    """

    def __init__(self, cfg: MTORConfig) -> None:
        self.cfg = cfg
        self._b: dict[str, float] = {}
        self._var: dict[str, float] = {}
        self._log_disc: dict[str, float] = {}
        self._log_disc_var: dict[str, float] = {}
        self._exposure: dict[str, int] = {}
        self._pairs: dict[tuple[str, int], tuple[tuple[int, ...], tuple[float, ...]]] = {}

    @property
    def pairs_cache(self) -> dict[tuple[str, int], tuple[tuple[int, ...], tuple[float, ...]]]:
        """Memo for ``MTOR._tag_pairs``, owned here rather than by ``MTOR``.

        An item's tag decomposition is a function of the item and the tag space,
        both of which outlive a single request -- but ``MTOR`` does not, so a memo
        on it would be thrown away before it was ever read twice. The item store
        is the nearest object whose lifetime matches the data, and the
        database-backed subclass widens it further to the whole process.
        """
        return self._pairs


    def ensure(self, item: Item) -> None:
        if item.id in self._b:
            return
        if item.difficulty_prior is None:
            self._b[item.id] = 0.0
            self._var[item.id] = self.cfg.item_prior_var
        else:
            d = _clip1(item.difficulty_prior, 0.0, 1.0)
            self._b[item.id] = (2.0 * d - 1.0) * self.cfg.difficulty_spread
            # A supplied prior is informative, so start tighter.
            self._var[item.id] = self.cfg.item_prior_var * 0.5
        self._log_disc[item.id] = 0.0          # slope 1.0
        self._log_disc_var[item.id] = self.cfg.disc_prior_var
        self._exposure[item.id] = 0

    def b(self, item: Item) -> float:
        self.ensure(item)
        return self._b[item.id]

    def var(self, item: Item) -> float:
        self.ensure(item)
        return self._var[item.id]

    def disc(self, item: Item) -> float:
        self.ensure(item)
        return math.exp(self._log_disc[item.id])

    def log_disc_var(self, item: Item) -> float:
        self.ensure(item)
        return self._log_disc_var[item.id]

    def exposure(self, item: Item) -> int:
        self.ensure(item)
        return self._exposure[item.id]

    def apply(self, item: Item, delta_b: float, new_var: float) -> None:
        self.ensure(item)
        self._b[item.id] += delta_b
        self._var[item.id] = max(new_var, self.cfg.min_var)
        self._exposure[item.id] += 1

    def apply_disc(self, item: Item, delta_log: float, new_var: float) -> None:
        self.ensure(item)
        clip = self.cfg.disc_log_clip
        self._log_disc[item.id] = _clip1(self._log_disc[item.id] + delta_log, -clip, clip)
        self._log_disc_var[item.id] = max(new_var, self.cfg.min_var)



class MTOR:
    """Implements ``contracts.core.Believe``, plus the ``predict`` half of ``Score``.

    ``predict`` lives here because the posterior update needs p_hat anyway;
    splitting it would mean computing the same quantity twice. The utility half
    of Score (rho / gamma*V) is a separate concern and lives elsewhere.
    """

    version = "mtor-1"

    def __init__(self, cfg: MTORConfig | None = None, items: ItemStore | None = None) -> None:
        self.cfg = cfg or MTORConfig()
        self.items = items if items is not None else ItemStore(self.cfg)

    # -- Believe ----------------------------------------------------------

    def init(self, user_id: str, space: TagSpace, prior: np.ndarray | None = None) -> Belief:
        n = space.n_dims
        mu = np.full(n, self.cfg.prior_mu, dtype=np.float64) if prior is None else np.asarray(prior, dtype=np.float64).copy()
        return Belief(
            user_id=user_id,
            mu=mu,
            var=np.full(n, self.cfg.prior_var, dtype=np.float64),
            # NaN means "never observed on this dimension". A 0.0 sentinel would
            # collide with a real timestamp (the Unix epoch), and the collision is
            # silent: the dimension would simply never inflate.
            last_seen=np.full(n, np.nan, dtype=np.float64),

            space=space,
            model_version=self.version,
        )

    def update(self, belief: Belief, signal: Signal, item: Item) -> Belief:
        idx, a = self._tag_vector(item, belief.space)
        if idx.size == 0:
            return belief

        r = _clip1(signal.outcome, 0.0, 1.0)
        p_hat, gz, c, s = self._predict_one(belief, item, idx, a)
        err = r - p_hat
        denom = max(p_hat * (1.0 - p_hat), 1e-9)

        # slope = d p_hat / d theta  per unit tag weight = (1 - c) * s(1-s) * gz
        # where gz = alpha * g folds in discrimination and uncertainty. With
        # alpha = 1, c = 0 this reduces to s(1-s)*g and the whole update below
        # collapses to the plain Rasch form -- no special-casing.
        slope = (1.0 - c) * s * (1.0 - s) * gz

        mu = belief.mu.copy()
        var = belief.var.copy()
        last_seen = belief.last_seen.copy()

        # natural-gradient mean step: mu += var * a * dLogLik/dtheta
        mu[idx] += var[idx] * a * (err * slope / denom)
        # Fisher information of the mean: (a*slope)^2 / (p(1-p))
        fisher = (a ** 2) * (slope ** 2) / denom
        # Fancy indexing returns a copy, so the floor has to be folded into the
        # assignment. ``np.maximum(var[idx], min_var, out=var[idx])`` computes
        # into a throwaway array and the clamp never lands in ``var``.
        var[idx] = np.maximum(1.0 / (1.0 / var[idx] + fisher), self.cfg.min_var)

        last_seen[idx] = signal.ts

        theta = float(np.dot(a, belief.mu[idx]))
        b = self.items.b(item)

        if self.cfg.learn_item_difficulty:
            b_var = self.items.var(item)
            # d p_hat / d b = -slope, so difficulty moves opposite to ability.
            self.items.apply(
                item,
                delta_b=-b_var * (err * slope / denom),
                new_var=1.0 / (1.0 / b_var + (slope ** 2) / denom),
            )

        if self.cfg.learn_discrimination and self.items.exposure(item) >= self.cfg.disc_min_exposure:
            # z = alpha*g*(theta-b); d z / d log(alpha) = z, so
            # d p_hat / d log(alpha) = (1-c) * s(1-s) * z.
            z = gz * (theta - b)
            dlog = (1.0 - c) * s * (1.0 - s) * z
            dv = self.items.log_disc_var(item)
            self.items.apply_disc(
                item,
                delta_log=dv * (err * dlog / denom),
                new_var=1.0 / (1.0 / dv + (dlog ** 2) / denom),
            )

        return replace(belief, mu=mu, var=var, last_seen=last_seen)



    def inflate(self, belief: Belief, now: float) -> Belief:
        seen = belief.last_seen
        days = np.where(np.isnan(seen), 0.0, (now - seen) / _SECONDS_PER_DAY)
        np.maximum(days, 0.0, out=days)


        drift = np.full(belief.space.n_dims, self.cfg.drift_per_day, dtype=np.float64)
        for tag, rate in self.cfg.drift_overrides.items():
            j = belief.space.index_of.get(tag)
            if j is not None:
                drift[j] = rate

        var = np.minimum(belief.var + (drift ** 2) * days, self.cfg.max_var)
        return replace(belief, var=var)

    # -- Score.predict ----------------------------------------------------

    def tag_layout(self, items: Sequence[Item], space: TagSpace) -> TagLayout:
        """Flatten a pool's tag incidence once. See ``TagLayout``."""
        tagged: list[int] = []
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        counts: list[int] = []
        for i, item in enumerate(items):
            c, v = self._tag_pairs(item, space)
            if not c:
                continue
            rows.extend([len(tagged)] * len(c))
            tagged.append(i)
            counts.append(len(c))
            cols.extend(c)
            vals.extend(v)
        counts_arr = np.asarray(counts, dtype=np.intp)
        starts = np.zeros(len(counts), dtype=np.intp)
        if len(counts) > 1:
            np.cumsum(counts_arr[:-1], out=starts[1:])
        return TagLayout(n=len(items),
                         tagged=np.asarray(tagged, dtype=np.intp),
                         rows=np.asarray(rows, dtype=np.intp),
                         cols=np.asarray(cols, dtype=np.intp),
                         vals=np.asarray(vals, dtype=np.float64),
                         starts=starts, counts=counts_arr)

    def predict(self, belief: Belief, items: Sequence[Item],
                layout: TagLayout | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised ``p_hat`` and ``sigma`` for a whole candidate pool.

        The scalar form of this (one ``_predict_one`` per item) measured as ~38%
        of a decide on a 2000-item pool, almost none of it arithmetic: it paid
        numpy's dispatch overhead 2000 times over for ``np.dot``/``np.clip`` on
        arrays of length 1-3. Here the (item, tag) pairs are reduced with
        ``bincount`` instead.

        Pass ``layout`` when the caller already built one for the same pool and
        tag space; otherwise it is built here.

        Untagged items keep the prior answer (0.5, sqrt(prior_var)) and are
        deliberately excluded from the gathers below: touching ``self.items`` for
        them would call ``ensure`` and mark a cold row dirty for an item this
        request never actually scored.
        """
        lay = layout if layout is not None else self.tag_layout(items, belief.space)
        p = np.full(lay.n, 0.5, dtype=np.float64)
        sigma = np.full(lay.n, math.sqrt(self.cfg.prior_var), dtype=np.float64)
        m = lay.m
        if m == 0:
            return p, sigma

        theta = np.bincount(lay.rows, weights=lay.vals * belief.mu[lay.cols], minlength=m)
        var_s = np.bincount(lay.rows,
                            weights=(lay.vals * lay.vals) * belief.var[lay.cols],
                            minlength=m)

        hot = [items[i] for i in lay.tagged]
        b = np.fromiter((self.items.b(it) for it in hot), np.float64, m)
        var_b = np.fromiter((self.items.var(it) for it in hot), np.float64, m)
        c = np.fromiter((self._floor(it) for it in hot), np.float64, m)
        alpha = (np.fromiter((self.items.disc(it) for it in hot), np.float64, m)
                 if self.cfg.learn_discrimination else 1.0)

        total_var = var_s + var_b
        g = 1.0 / np.sqrt(1.0 + LOGIT_VAR_CORRECTION * total_var)
        s = _sigmoid(alpha * g * (theta - b))
        p[lay.tagged] = c + (1.0 - c) * s
        sigma[lay.tagged] = np.sqrt(total_var)
        return p, sigma

    def scored(self, belief: Belief, items: Sequence[Item],
               layout: TagLayout | None = None) -> list[Scored]:
        p, s = self.predict(belief, items, layout)
        return [Scored(item_id=it.id, p_hat=float(p[i]), sigma=float(s[i]), utility=0.0)
                for i, it in enumerate(items)]

    def fisher_information(self, belief: Belief, items: Sequence[Item],
                           layout: TagLayout | None = None) -> np.ndarray:
        """p(1-p), maximal at p = 0.5.

        This is the whole of the cold-start story: "pick what you are least
        certain about" is not a separate strategy bolted on, it is the argmax of
        this quantity. Adaptive testing falls out of the same formula.
        """
        p, _ = self.predict(belief, items, layout)
        return p * (1.0 - p)

    # -- internals --------------------------------------------------------

    def _tag_pairs(self, item: Item, space: TagSpace) -> tuple[Sequence[int], Sequence[float]]:
        """Resolve an item's tags into (dimension indices, normalised weights).

        Plain Python sequences, not arrays. This is the single definition of the
        resolution; ``_tag_vector`` wraps it for the single-item callers and
        ``tag_layout`` consumes it directly. Building two length-1-to-3 numpy
        arrays here only to call ``.tolist()`` on them a moment later was 8000
        wasted array constructions per decide -- at that size ``np.fromiter``
        costs far more than the arithmetic it carries.

        The result is memoised on the item store, because nothing in it depends on
        the request: an item's tags change only when the item is upserted, and tag
        indices are append-only, so the number of dimensions is enough to detect a
        space that has grown. Tuples, so a cached entry cannot be mutated by a
        caller that got it from the memo instead of computing it.

        An untagged item routes to the reserved latent dimension when present,
        which is how the model keeps working for customers who supply no
        taxonomy at all.
        """
        index_of = space.index_of
        cache = self.items.pairs_cache
        key = (item.id, len(index_of))
        hit = cache.get(key)
        if hit is not None:
            return hit

        cols: list[int] = []
        vals: list[float] = []
        total = 0.0
        for t, w in item.tag_weights.items():
            j = index_of.get(t)
            if j is None or w <= 0.0:
                continue
            w = float(w)
            cols.append(j)
            vals.append(w)
            total += w
        if not cols:
            j = index_of.get(TagSpace.LATENT)
            out = ((), ()) if j is None else ((j,), (1.0,))
        else:
            if total > 0.0:
                vals = [v / total for v in vals]
            out = (tuple(cols), tuple(vals))
        cache[key] = out
        return out


    def _tag_vector(self, item: Item, space: TagSpace) -> tuple[np.ndarray, np.ndarray]:
        """Array form of ``_tag_pairs``, for the single-item update path."""
        cols, vals = self._tag_pairs(item, space)
        return (np.asarray(cols, dtype=np.intp), np.asarray(vals, dtype=np.float64))

    def _floor(self, item: Item) -> float:
        """Baseline success probability from item metadata (0.0 if unspecified).

        A non-finite attribute is treated as unspecified. ``min``/``max`` propagate
        NaN rather than clamping it (every NaN comparison is False), and a NaN
        floor turns ``p_hat`` into NaN, which then poisons the ADF update for every
        user who touches the item without raising anything.
        """
        if self.cfg.floor_attr is None:
            return 0.0
        v = item.attrs.get(self.cfg.floor_attr)
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(f):
            return 0.0
        return min(max(f, 0.0), 0.95)


    def _predict_one(
        self, belief: Belief, item: Item, idx: np.ndarray, a: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Return (p_hat, gz, c, s) where

            z      = alpha * g * (theta - b)      discrimination-scaled logit
            s      = sigmoid(z)
            p_hat  = c + (1 - c) * s
            gz     = alpha * g                    d z / d theta, reused by update

        With alpha = 1 (slope learning off) and c = 0 this is the plain Rasch
        form, so both extensions are strictly additive -- callers that use
        neither see identical behaviour.
        """
        theta = float(np.dot(a, belief.mu[idx]))
        var_s = float(np.sum((a ** 2) * belief.var[idx]))
        b = self.items.b(item)
        var_b = self.items.var(item)
        g = 1.0 / math.sqrt(1.0 + LOGIT_VAR_CORRECTION * (var_s + var_b))
        alpha = self.items.disc(item) if self.cfg.learn_discrimination else 1.0
        gz = alpha * g
        s = float(_sigmoid(gz * (theta - b)))
        c = self._floor(item)
        return c + (1.0 - c) * s, gz, c, s


