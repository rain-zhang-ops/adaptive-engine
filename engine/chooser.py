"""Choose -- constrained subset selection.

    argmax_{A in F, |A| = k}  U(A | b)

Exact optimisation over a combinatorial action space under partial
observability is intractable, so the sanctioned approximation is greedy
selection over a submodular objective, which carries a (1 - 1/e) guarantee
while Phi stays submodular. That bound is part of the contract and is stated
rather than hidden -- and it is why ``StructureSpec.weight`` must be
non-negative: a sign flip destroys submodularity and the guarantee with it.

The scope of that guarantee is stated just as plainly: of the shipped Phi
kinds, only ``diversify`` (negative max-similarity) has a non-increasing
marginal in general. ``concentrate`` (mean similarity) and ``balanced``
(distance to a target entropy) do not -- their marginals can rise as the set
grows -- so for those kinds greedy selection is a heuristic and no
approximation bound is claimed. The submodularity invariant test therefore
measures ``diversify`` only.

Three things here are easy to get wrong and are handled explicitly:

*Hard constraints are filters, not penalties.* Anything that must never be
returned regardless of score (an embargoed item, another tenant's catalogue) is
removed during recall. Encoding such a rule as a score penalty means a
sufficiently attractive item eventually leaks through.

*Quotas must be satisfiable, not merely preferred.* Greedy selection that
ignores remaining quota happily fills every slot with the highest-scoring group
and then cannot meet the rest. Admissibility therefore accounts for how many
slots are still needed by under-filled groups.

*Exploration needs a propensity.* Interactions are never a random sample; they
are whatever the policy chose. Without recording the selection probability
there is no way to correct that bias later, and it cannot be reconstructed
after the fact. Exploration slots sample from a known distribution precisely so
this number exists.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from contracts.core import (
    Chosen,
    Constraints,
    Decision,
    Item,
    Scored,
    StructureSpec,
    Utility,
)
from engine.predicates import Predicate, compile_all

__all__ = ["GreedyChooser", "n_explore_for"]


def n_explore_for(explore_floor: float, k: int) -> int:
    """How many of ``k`` slots are reserved for exploration.

    ``round`` alone collapses the floor to zero for small slates -- a policy with
    ``explore_floor=0.10`` explores nothing at ``k<=4``, so the anti-filter-bubble
    guarantee silently lapses exactly where a caller is least likely to notice.
    A positive floor therefore reserves at least one slot once ``k`` allows it,
    while never consuming every slot.
    """
    if k <= 1 or explore_floor <= 0.0:
        return 0
    n = int(round(explore_floor * k))
    if n == 0:
        n = 1
    return min(n, k - 1)



def _get_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _similarity(a: Item, b: Item, kind: str) -> float:
    ta, tb = a.tag_weights, b.tag_weights
    if not ta or not tb:
        # Untagged items are treated as maximally interchangeable: with no
        # taxonomy there is no evidence they differ, and pretending otherwise
        # would let a "diversify" policy stack near-duplicates.
        return 1.0 if not ta and not tb else 0.0
    keys = set(ta) | set(tb)
    va = np.array([ta.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([tb.get(k, 0.0) for k in keys], dtype=float)
    if kind == "tag_cosine":
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else 0.0
    inter = float(np.minimum(va, vb).sum())
    union = float(np.maximum(va, vb).sum())
    return inter / union if union > 0 else 0.0


def _tag_entropy(items: Sequence[Item]) -> float:
    mass: dict[str, float] = {}
    for it in items:
        for t, w in it.tag_weights.items():
            mass[t] = mass.get(t, 0.0) + float(w)
    total = sum(mass.values())
    if total <= 0:
        return 0.0
    p = np.array(list(mass.values()), dtype=float) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class _TagSpace:
    """Dense tag matrix over one recalled pool, built once per solve.

    Greedy selection asks for the structure marginal of *every* remaining
    candidate at *every* step, so the pairwise similarity is evaluated O(k^2 * n)
    times. Doing that as a Python loop over tag dicts was measured at ~80% of
    total decide time (~360 ms for k=10 over a 2000-item pool). The fix is not a
    different algorithm -- the selection is unchanged -- it is evaluating the same
    quantity for all candidates at once.

    The dense layout is affordable because ``d`` is the pool's tag vocabulary,
    which is the same order as the belief dimension, not the catalogue size.
    """

    __slots__ = ("kind", "M", "n", "d", "has", "norms", "row_sum")

    def __init__(self, pool: Sequence[Item], kind: str) -> None:
        self.kind = kind
        vocab: dict[str, int] = {}
        for it in pool:
            for t in it.tag_weights:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.n = len(pool)
        self.d = len(vocab)
        self.M = np.zeros((self.n, self.d), dtype=np.float64)
        for i, it in enumerate(pool):
            for t, w in it.tag_weights.items():
                self.M[i, vocab[t]] = float(w)
        # "untagged" is an empty tag map, not a zero row: an item tagged {a: 0.0}
        # has a taxonomy entry and must not be treated as interchangeable.
        self.has = np.array([bool(it.tag_weights) for it in pool], dtype=bool)
        self.norms = np.linalg.norm(self.M, axis=1)
        self.row_sum = self.M.sum(axis=1)

    def sims_to(self, j: int) -> np.ndarray:
        """Similarity of every pool item to pool item ``j``."""
        if not self.has[j]:
            # matches the scalar rule: two untagged items are maximally
            # interchangeable, one untagged and one tagged share nothing
            return np.where(self.has, 0.0, 1.0)
        v = self.M[j]
        if self.kind == "tag_cosine":
            denom = self.norms * self.norms[j]
            out = np.zeros(self.n, dtype=np.float64)
            ok = denom > 0
            out[ok] = (self.M[ok] @ v) / denom[ok]
        else:
            inter = np.minimum(self.M, v).sum(axis=1)
            union = np.maximum(self.M, v).sum(axis=1)
            out = np.zeros(self.n, dtype=np.float64)
            ok = union > 0
            out[ok] = inter[ok] / union[ok]
        return np.where(self.has, out, 0.0)


class _StructureState:
    """Incremental Phi bookkeeping for one greedy run.

    ``max``/``sum`` over the selected set are maintained rather than recomputed:
    both are updated by one vector operation per committed item, which drops the
    similarity work from O(k^2 * n) to O(k * n) on top of the vectorisation.
    """

    __slots__ = ("space", "spec", "max_sim", "sum_sim", "n_sel", "mass", "total")

    def __init__(self, space: _TagSpace, spec: StructureSpec) -> None:
        self.space = space
        self.spec = spec
        self.max_sim = np.zeros(space.n, dtype=np.float64)
        self.sum_sim = np.zeros(space.n, dtype=np.float64)
        self.n_sel = 0
        self.mass = np.zeros(space.d, dtype=np.float64)
        self.total = 0.0

    def add(self, j: int) -> None:
        sims = self.space.sims_to(j)
        np.maximum(self.max_sim, sims, out=self.max_sim)
        self.sum_sim += sims
        self.n_sel += 1
        self.mass += self.space.M[j]
        self.total += float(self.space.row_sum[j])

    def marginals(self) -> np.ndarray:
        w = self.spec.weight
        if self.n_sel == 0:
            # An empty set has no structure to trade against; the first pick is
            # decided by fit alone, exactly as in the scalar reference.
            return np.zeros(self.space.n, dtype=np.float64)
        if self.spec.kind == "diversify":
            return -w * self.max_sim
        if self.spec.kind == "concentrate":
            return w * (self.sum_sim / self.n_sel)
        if self.spec.kind == "balanced":
            target = self.spec.target_entropy or 0.0
            before = abs(_entropy_of(self.mass, self.total) - target)
            after = np.abs(self._entropy_with_each() - target)
            return w * (before - after)
        raise ValueError(f"unknown structure kind {self.spec.kind!r}")

    def _entropy_with_each(self) -> np.ndarray:
        mass = self.mass + self.space.M                     # (n, d)
        total = self.total + self.space.row_sum             # (n,)
        out = np.zeros(self.space.n, dtype=np.float64)
        ok = total > 0
        if not np.any(ok):
            return out
        p = mass[ok] / total[ok][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p), 0.0)
        out[ok] = -terms.sum(axis=1)
        return out


def _entropy_of(mass: np.ndarray, total: float) -> float:
    if total <= 0:
        return 0.0
    p = mass / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())



class GreedyChooser:
    """Implements ``contracts.core.Choose``."""

    def __init__(self, seed: int = 0, explore_pool_factor: int = 3) -> None:
        self.rng = np.random.default_rng(seed)
        self.explore_pool_factor = explore_pool_factor
        """Exploration samples uniformly from the top (factor * slots) items by
        posterior sigma. A fixed, known pool size is what makes the propensity
        exact instead of an estimate."""

    # -- Phi --------------------------------------------------------------

    def _structure_marginal(
        self, cand: Item, selected: list[Item], spec: StructureSpec | None
    ) -> float:
        """Scalar reference implementation of Phi's marginal.

        Kept even though the hot path is vectorised: it is the definition, it is
        what the submodularity test measures, and it is the oracle the fast path
        is checked against. Two implementations that must agree is a cheaper
        safeguard than one implementation nobody can read.
        """
        if spec is None or spec.weight == 0.0 or not selected:
            return 0.0
        sims = [_similarity(cand, s, spec.similarity) for s in selected]
        if spec.kind == "diversify":
            # Penalise the *nearest* already-selected item. Max, not mean:
            # what hurts a varied set is a near-duplicate, and averaging lets
            # one duplicate hide behind many dissimilar picks.
            return -spec.weight * max(sims)
        if spec.kind == "concentrate":
            # Reward overall cohesion. Mean, not max: the goal is a set that
            # hangs together, not one that merely touches the cluster.
            return spec.weight * (sum(sims) / len(sims))
        if spec.kind == "balanced":
            target = spec.target_entropy or 0.0
            before = abs(_tag_entropy(selected) - target)
            after = abs(_tag_entropy(selected + [cand]) - target)
            return spec.weight * (before - after)
        raise ValueError(f"unknown structure kind {spec.kind!r}")

    @staticmethod
    def _structure_state(pool: Sequence[Item], spec: StructureSpec | None):
        if spec is None or spec.weight == 0.0:
            return None
        return _StructureState(_TagSpace(pool, spec.similarity), spec)

    def _best_admissible(
        self,
        pool: Sequence[Item],
        gain: np.ndarray,
        constraints: Constraints,
        filled: dict,
        tag_count: dict,
        slots_left: int,
        needs_admission: bool,
    ) -> int | None:
        """Highest-gain candidate that is still admissible, or None.

        ``gain`` carries ``-inf`` for already-taken items. When no quota or tag
        cap is configured -- the common case -- this is a single argmax; when one
        is, candidates are walked in descending gain and the first admissible one
        wins, which is the same answer the per-item scan gave. Stable sorting
        keeps the original pool-order tie-break.

        A NaN gain is demoted to ``-inf`` rather than allowed to win the argmax.
        One bad utility (an upstream belief with a NaN dimension) would otherwise
        be returned by ``argmax``, fail the finiteness check, and abort the whole
        slate -- turning a single unusable candidate into a batch-wide fallback.
        """
        gain = np.where(np.isnan(gain), -np.inf, gain)
        if not needs_admission:

            i = int(np.argmax(gain))
            return i if np.isfinite(gain[i]) else None
        for i in np.argsort(-gain, kind="stable"):
            i = int(i)
            if not np.isfinite(gain[i]):
                return None
            it = pool[i]
            if not self._quota_admissible(it, constraints, filled, slots_left):
                continue
            if not self._tag_admissible(it, constraints, tag_count):
                continue
            return i
        return None


    # -- constraints ------------------------------------------------------

    def _hard_filter(
        self, items: Sequence[Item], scored: Sequence[Scored], constraints: Constraints
    ) -> tuple[list[Item], list[Scored]]:
        preds: list[Predicate] = compile_all(constraints.predicates)
        within = constraints.within_tags
        keep_i, keep_s = [], []
        by_id = {s.item_id: s for s in scored}
        seen: set[str] = set()
        for it in items:
            if it.id in constraints.exclude_item_ids:
                continue
            if it.id in seen:
                # Everything below keys candidates by id, so a pool holding the
                # same id twice would leave one index selectable after the other
                # was committed -- the same item twice in one slate.
                continue
            if within and not (set(it.tag_weights) & within):
                continue
            if any(not p.holds(it) for p in preds):
                continue
            s = by_id.get(it.id)
            if s is None:
                continue
            seen.add(it.id)
            keep_i.append(it)
            keep_s.append(s)
        return keep_i, keep_s


    @staticmethod
    def _group_of(item: Item, path: str) -> str:
        return str(_get_path(item, path))

    def _quota_admissible(
        self, item: Item, constraints: Constraints, filled: dict, slots_left: int
    ) -> bool:
        """Admissible if the item's group still needs slots, or enough slack
        remains that taking it cannot starve an under-filled group."""
        for q in constraints.quotas:
            g = self._group_of(item, q.group_by)
            need = {k: max(0, v - filled[q.group_by].get(k, 0)) for k, v in q.counts.items()}
            unmet = sum(need.values())
            if need.get(g, 0) > 0:
                continue
            if slots_left <= unmet:
                return False
        return True

    def _tag_admissible(self, item: Item, constraints: Constraints, tag_count: dict) -> bool:
        if constraints.max_per_tag is None:
            return True
        return all(tag_count.get(t, 0) < constraints.max_per_tag for t in item.tag_weights)

    # -- Choose.solve -----------------------------------------------------

    def solve(
        self,
        scored: Sequence[Scored],
        items: Sequence[Item],
        utility: Utility,
        constraints: Constraints,
    ) -> Decision:
        if not items:
            return Decision(chosen=(), confidence="low", fallback_reason="empty_candidate_pool")

        pool, pool_scored = self._hard_filter(items, scored, constraints)
        if not pool:
            return Decision(chosen=(), confidence="low",
                            fallback_reason="constraints_unsatisfiable")

        base = {s.item_id: s for s in pool_scored}
        k = min(constraints.k, len(pool))

        n_explore = n_explore_for(utility.explore_floor, k)
        n_exploit = k - n_explore


        filled = {q.group_by: {} for q in constraints.quotas}
        tag_count: dict[str, int] = {}
        selected: list[Item] = []
        chosen: list[Chosen] = []
        taken: set[str] = set()

        idx_of = {it.id: i for i, it in enumerate(pool)}
        util_vec = np.array([base[it.id].utility for it in pool], dtype=np.float64)
        state = self._structure_state(pool, utility.structure)
        alive = np.ones(len(pool), dtype=bool)
        needs_admission = bool(constraints.quotas) or constraints.max_per_tag is not None

        def commit(item: Item, marginal: float, structure: float, propensity: float,
                   explore: bool, exact: bool = True) -> None:
            s = base[item.id]
            chosen.append(Chosen(
                item_id=item.id,
                utility=float(s.utility + structure),
                p_hat=float(s.p_hat),
                marginal_gain=float(marginal),
                propensity=float(propensity),
                reasons={"fit": float(s.utility), "structure": float(structure),
                         "sigma": float(s.sigma), "explore": 1.0 if explore else 0.0,
                         "propensity_exact": 1.0 if exact else 0.0},
            ))

            selected.append(item)
            taken.add(item.id)
            j = idx_of[item.id]
            alive[j] = False
            if state is not None:
                state.add(j)
            for q in constraints.quotas:
                g = self._group_of(item, q.group_by)
                filled[q.group_by][g] = filled[q.group_by].get(g, 0) + 1
            for t in item.tag_weights:
                tag_count[t] = tag_count.get(t, 0) + 1

        def structure_vector() -> np.ndarray:
            return (state.marginals() if state is not None
                    else np.zeros(len(pool), dtype=np.float64))

        # -- exploit: deterministic greedy over rho + gamma*V + Phi --------
        for _ in range(n_exploit):
            slots_left = k - len(selected)
            struct_vec = structure_vector()
            gain = np.where(alive, util_vec + struct_vec, -np.inf)
            i = self._best_admissible(pool, gain, constraints, filled, tag_count,
                                      slots_left, needs_admission)
            if i is None:
                break
            commit(pool[i], float(gain[i]), float(struct_vec[i]),
                   propensity=1.0, explore=False)


        # -- explore: uniform WITHOUT replacement over a pool fixed up front.
        #
        # The pool is built once and the propensity reported is the *marginal*
        # inclusion probability n_explore / m, not the per-draw conditional
        # 1 / m. Those differ by a factor of n_explore, and since the number is
        # consumed as an inverse-propensity weight, using the conditional would
        # scale every off-policy estimate by that factor -- a bias that looks
        # like a result. Fixing the pool before drawing is what makes the
        # marginal exactly computable instead of merely estimated.
        if n_explore > 0:
            avail = [it for it in pool
                     if it.id not in taken
                     and self._quota_admissible(it, constraints, filled, n_explore)
                     and self._tag_admissible(it, constraints, tag_count)]
            avail.sort(key=lambda it: base[it.id].sigma, reverse=True)
            m = min(len(avail), max(1, self.explore_pool_factor * n_explore))
            top = avail[:m]
            propensity = min(1.0, n_explore / m) if m else 0.0

            order = self.rng.permutation(m)
            exact = True
            for pos in order:
                if len(selected) - n_exploit >= n_explore:
                    break
                pick = top[int(pos)]
                slots_left = k - len(selected)
                if not (self._quota_admissible(pick, constraints, filled, slots_left)
                        and self._tag_admissible(pick, constraints, tag_count)):
                    # A rejection mid-draw makes the marginal above an
                    # approximation. Flagged rather than silently reported as
                    # exact: a caller doing IPS needs to know which rows to trust.
                    exact = False
                    continue
                struct_vec = structure_vector()
                j = idx_of[pick.id]
                struct = float(struct_vec[j])
                commit(pick, base[pick.id].utility + struct, struct,
                       propensity=propensity, explore=True, exact=exact)



        unmet = self._unmet_quota(constraints, filled)
        if len(chosen) < constraints.k or unmet:
            # Three distinct causes, three distinct reasons. Collapsing "the
            # catalogue is empty" into "the catalogue is smaller than k" sends
            # whoever is debugging after a problem that does not exist.
            if unmet:
                reason = "constraints_unsatisfiable"
            elif not chosen:
                reason = "empty_candidate_pool"
            else:
                reason = "insufficient_candidates"
            return Decision(chosen=tuple(chosen), confidence="low", fallback_reason=reason,
                            model_version="", policy_id="")


        return Decision(chosen=tuple(chosen), confidence="high", fallback_reason=None)

    @staticmethod
    def _unmet_quota(constraints: Constraints, filled: dict) -> bool:
        for q in constraints.quotas:
            for group, want in q.counts.items():
                if filled[q.group_by].get(group, 0) < want:
                    return True
        return False

    # -- inclusion probabilities (for off-policy evaluation) --------------

    def inclusion_probabilities(
        self,
        scored: Sequence[Scored],
        items: Sequence[Item],
        utility: Utility,
        constraints: Constraints,
    ) -> dict[str, float]:
        """P(item in the returned set) for every candidate, computed analytically.

        This exists so off-policy evaluation does not have to estimate the target
        policy's action distribution by sampling. It can be exact here because
        the policy has exactly two parts: a deterministic greedy phase (probability
        1 or 0) and a uniform-without-replacement draw from a pool fixed before any
        randomness is consumed (probability n_explore / m for every pool member,
        including the ones this particular draw did not pick).

        The subtlety worth being explicit about: an explore-pool member that was
        *not* drawn still had probability n_explore / m of being drawn. A naive
        implementation that reads probabilities off one realised decision would
        assign it zero, silently shrinking the target policy's support and biasing
        the IPS estimate toward whatever the logging policy happened to do.
        """
        probs: dict[str, float] = {it.id: 0.0 for it in items}
        if not items:
            return probs

        pool, pool_scored = self._hard_filter(items, scored, constraints)
        if not pool:
            return probs

        base = {s.item_id: s for s in pool_scored}
        k = min(constraints.k, len(pool))
        n_explore = n_explore_for(utility.explore_floor, k)
        n_exploit = k - n_explore


        filled = {q.group_by: {} for q in constraints.quotas}
        tag_count: dict[str, int] = {}
        selected: list[Item] = []
        taken: set[str] = set()

        for _ in range(n_exploit):
            slots_left = k - len(selected)
            best, best_gain = None, -math.inf
            for it in pool:
                if it.id in taken:
                    continue
                if not self._quota_admissible(it, constraints, filled, slots_left):
                    continue
                if not self._tag_admissible(it, constraints, tag_count):
                    continue
                gain = base[it.id].utility + self._structure_marginal(
                    it, selected, utility.structure)
                if gain > best_gain:
                    best, best_gain = it, gain
            if best is None:
                break
            probs[best.id] = 1.0
            selected.append(best)
            taken.add(best.id)
            for q in constraints.quotas:
                g = self._group_of(best, q.group_by)
                filled[q.group_by][g] = filled[q.group_by].get(g, 0) + 1
            for t in best.tag_weights:
                tag_count[t] = tag_count.get(t, 0) + 1

        if n_explore > 0:
            avail = [it for it in pool
                     if it.id not in taken
                     and self._quota_admissible(it, constraints, filled, n_explore)
                     and self._tag_admissible(it, constraints, tag_count)]
            avail.sort(key=lambda it: base[it.id].sigma, reverse=True)
            m = min(len(avail), max(1, self.explore_pool_factor * n_explore))
            if m:
                p = min(1.0, n_explore / m)
                for it in avail[:m]:
                    probs[it.id] = p
        return probs

