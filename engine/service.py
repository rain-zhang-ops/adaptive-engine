"""EngineService -- the one object the HTTP layer talks to.

This is where persistence, tenant isolation, versioning and the three primitives
compose into the two operations a caller actually performs: *decide* (ask for
items) and *observe* (report what happened). Everything below the API is
assembled here so the transport layer stays a thin translation of HTTP to these
two calls.

Design commitments, each answering a specific production failure:

*Degrade, never raise.* A recommendation engine that returns 5xx turns its own
hiccup into the caller's outage. Unknown user, empty candidates, thin data --
all resolve to a Decision with ``confidence`` lowered and a ``fallback_reason``
set. The only exceptions that escape are programmer errors (a policy that fails
its own load-time validation), and those surface at deploy, not at request.

*Every decision is reproducible.* ``model_version``, ``policy_id`` and a content
hash of the goal+tune are stamped onto the Decision and logged. "Why did the
engine pick this?" must be answerable months later from the audit row alone.

*Observation is idempotent and serialised per user.* Belief update is
read-modify-write; the whole of it runs inside one ``BEGIN IMMEDIATE`` together
with the signal-id claim, so a duplicate delivery is a no-op and two concurrent
signals for one user cannot lose an update.

*inflate on read.* Time-based uncertainty growth is applied when a belief is
loaded, using the request clock, rather than by a background sweep. A belief is
only ever as stale as its last read, and there is no cron job to operate.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from contracts.core import Belief, Chosen, Constraints, Decision, Item, Signal
from engine.chooser import GreedyChooser, n_explore_for

from engine.mtor import MTOR, MTORConfig
from engine.observability import Metrics, Timer, log_event
from engine.policy import (
    GoalCatalog,
    PolicyError,
    constraints_from,
    load_catalog,
    validate_policy_doc,
)

from engine.scorer import UtilityScorer
from engine.store import SqliteStore, TenantItemStore


__all__ = ["EngineService", "DecideResult", "ObserveResult", "ServiceConfig"]


@dataclass(frozen=True)
class ServiceConfig:
    mtor: MTORConfig = field(default_factory=lambda: MTORConfig(floor_attr="floor"))
    recall_limit: int = 800
    """Ceiling on candidates pulled from store per decision. Also the cap that
    stops one pathological request from scanning an entire tenant catalogue.

    800 rather than a rounder 2000 because 2000 was asserted to be "far past where
    quality saturates" and measurement put the saturation point lower. Priced
    through ``decide`` against the simulator's ``p_true`` (the same ground truth
    ``engine.ope`` validates against), 3 catalogue seeds x 120 warmed users x 2
    goals, baseline = the untruncated catalogue:

    * 800 -- p50 24 -> 11ms, and every quality measure inside 1 SE of no
      truncation at all (chosen-slate utility within +/-0.9%, expected successes
      within +/-0.7%)
    * 300 -- utility -3.8..-4.4%, expected successes -4.7..-8.0%, consistent
      across all three seeds and ~10x the standard error. A real trade, available
      to anyone who wants it, but not a default.

    Read the arms as "where does the loss start", not as a ranking: pools across
    arms are not nested, since recall is an objective-driven branch plus an
    exploration sample and the reserve draws from a different set at each limit.
    """


    recall_tags: int = 12
    """How many top-weighted tags drive objective-driven recall. Wide enough that
    a multi-tag catalogue still yields a varied pool, narrow enough that the query
    stays selective -- past this the tag filter stops filtering."""

    explore_pool_factor: int = 3


@dataclass(frozen=True)
class DecideResult:
    decision: Decision
    decision_id: str
    goal: str
    applied: Mapping[str, Any]
    notes: Sequence[str]
    warnings: Sequence[str]
    recall: Mapping[str, Any] = field(default_factory=dict)
    """How the candidate pool was built. Surfaced rather than hidden because
    "why wasn't item X considered?" is otherwise unanswerable, and a truncated
    catalogue is exactly the condition a caller needs to know about."""



@dataclass(frozen=True)
class ObserveResult:
    accepted: int
    duplicates: int
    unknown_items: int
    backfilled_propensity: int = 0
    """Signals that carried no propensity but referenced a decision_id we could
    look it up from. Reported so a caller can see the loop is closing itself."""

    missing_propensity: int = 0
    """Accepted signals with neither a propensity nor a resolvable decision_id.
    These still train the belief, but they are useless for off-policy evaluation
    -- counted rather than dropped so the gap is visible instead of silent."""




def _policy_hash(goal: str, tune: Mapping[str, Any] | None) -> str:
    blob = json.dumps({"goal": goal, "tune": tune or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _merge_constraints(base: Mapping[str, Any] | None,
                       over: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Request constraints override policy constraints, key by key.

    An L3 policy can ship a feasible set (embargo predicates, a kind quota), and
    a single request can still tighten it further. Per-key override, not deep
    merge, so a request that sets ``quotas`` replaces the policy's quotas wholesale
    rather than silently concatenating two quota lists that may contradict.
    """
    if not base:
        return dict(over) if over else None
    if not over:
        return dict(base)
    return {**base, **over}



class EngineService:
    def __init__(self, store: SqliteStore, catalog: GoalCatalog | None = None,
                 cfg: ServiceConfig | None = None, metrics: Metrics | None = None) -> None:
        self.store = store
        self.cfg = cfg or ServiceConfig()
        self.catalog = catalog or load_catalog()
        self.metrics = metrics or Metrics()
        self.model_version = MTOR(self.cfg.mtor).version


    # -- helpers ----------------------------------------------------------

    def _mtor_for(self, tenant: str, con=None) -> tuple[MTOR, TenantItemStore]:
        items = TenantItemStore(self.cfg.mtor, self.store, tenant, con=con)
        return MTOR(self.cfg.mtor, items), items

    def _load_belief(self, mtor: MTOR, tenant: str, user_id: str, now: float,
                     con=None) -> tuple[Belief, bool]:
        space = self.store.tag_space(tenant)
        raw = self.store.load_raw_belief(tenant, user_id, con=con)
        if raw is None:
            return mtor.init(user_id, space), False
        # A belief persisted before new tags were registered is short some
        # dimensions; pad to the current space so a growing taxonomy never
        # invalidates stored state.
        n = space.n_dims
        mu = _fit(raw.mu, n, self.cfg.mtor.prior_mu)
        var = _fit(raw.var, n, self.cfg.mtor.prior_var)
        last_seen = _fit(raw.last_seen, n, np.nan)

        b = Belief(user_id=user_id, mu=mu, var=var, last_seen=last_seen,
                   space=space, model_version=raw.model_version)
        return mtor.inflate(b, now), True

    # -- recall -----------------------------------------------------------

    def _recall(self, tenant: str, scorer: UtilityScorer, belief, utility, k: int,
                seed: int, within_tags: Sequence[str] | None = None,
                ) -> tuple[list[Item], dict[str, Any]]:

        """Candidate generation.

        Two slices, and both are necessary:

        *Objective-driven* -- items carrying the tags this policy actually cares
        about, taken from the tag index. Which tags those are comes from the same
        weighting the scorer uses, so recall and scoring cannot disagree about the
        objective.

        *Coverage* -- a deterministic pseudo-random slice. Objective-driven recall
        alone is a closed loop: it surfaces only what the current belief already
        favours, so anything outside that neighbourhood never accrues exposure and
        the model can never discover it was wrong about it. The slice is sized by
        what the chooser downstream will actually consume for exploration
        (``explore_pool_factor * n_explore``), not by a number picked for feel.

        If the catalogue fits inside the budget, both slices are moot and the
        whole catalogue is the pool -- which is the common case for a new tenant
        and keeps behaviour simple to reason about.
        """
        budget = self.cfg.recall_limit
        if within_tags:
            # A tag-scoped request is answered from the tag index only. Widening
            # the pool past the scope would just hand the chooser items its hard
            # filter must drop, and the coverage slice below has no meaning
            # inside a scope the caller picked deliberately.
            pool = _dedupe_items(
                self.store.recall_by_tags(tenant, list(within_tags), budget))
            return pool, {"strategy": "within_tags", "scanned": len(pool),
                          "catalogue": self.store.item_count(tenant),
                          "truncated": len(pool) >= budget,
                          "tags": list(within_tags)[:8]}

        total_items = self.store.item_count(tenant)
        if total_items <= budget:
            pool = self.store.get_items(tenant, None, limit=budget)
            return pool, {"strategy": "full_catalogue", "scanned": len(pool),
                          "catalogue": total_items, "truncated": False}

        n_explore = n_explore_for(utility.explore_floor, k)
        min_cov = self.cfg.explore_pool_factor * max(n_explore, 1)
        n_cov = min(budget - k, max(min_cov, int(budget * utility.explore_floor)))
        n_obj = budget - n_cov

        space = belief.space
        w = scorer.recall_weights(belief, utility)
        order = np.argsort(-w)
        tags = [space.tag_of[j] for j in order
                if space.tag_of[j] != "_latent"][:self.cfg.recall_tags]

        objective = self.store.recall_by_tags(tenant, tags, n_obj)
        seen = [it.id for it in objective]
        coverage = self.store.sample_items(tenant, n_cov, seed=seed, exclude=seen)

        # De-duplicate. The two slices are drawn independently, and an id present
        # in both would reach the chooser twice: its id->index map keeps only the
        # last occurrence, so committing the item leaves the earlier index still
        # selectable and the same item can be returned twice in one slate.
        pool = _dedupe_items(objective + coverage)
        return pool, {"strategy": "tag_index+coverage", "scanned": len(pool),
                      "catalogue": total_items, "truncated": True,
                      "objective": len(objective), "coverage": len(coverage),
                      "duplicates_dropped": len(objective) + len(coverage) - len(pool),
                      "tags": tags[:8]}


    # -- policy resolution -------------------------------------------------

    def _resolve_policy(self, tenant: str, goal: str | None,
                        tune: Mapping[str, Any] | None,
                        policy_ref: str | None):
        """(Resolved, policy_id, policy constraints) for L0/L1/L2 or L3.

        L3 goes through the same ``resolve_doc`` validation as the shipped
        catalogue, so a tenant policy cannot express something the engine would
        have refused to load. ``policy_id`` becomes the ref itself, which is more
        useful in an audit row than a hash of a document stored elsewhere.
        """
        if policy_ref:
            row = self.store.get_policy(tenant, policy_ref)
            if row is None:
                known = [p["policy_ref"] for p in self.store.list_policies(tenant)]
                raise PolicyError(
                    f"unknown policy_ref {policy_ref!r}; registered: {sorted(known)}")
            resolved, cons = self.catalog.resolve_doc(row["doc"], tune)
            return resolved, policy_ref, cons
        resolved = self.catalog.resolve(goal, tune)   # PolicyError only on misconfig
        return resolved, _policy_hash(resolved.goal, tune), None

    # -- decide -----------------------------------------------------------

    def decide_many(
        self,
        tenant: str,
        user_ids: Sequence[str],
        count: int,
        goal: str | None = None,
        tune: Mapping[str, Any] | None = None,
        constraints_spec: Mapping[str, Any] | None = None,
        candidate_ids: Sequence[str] | None = None,
        now: float = 0.0,
        policy_ref: str | None = None,
    ) -> list[DecideResult]:
        """Batch decide.

        Recall is belief-driven, so pools are per user and cannot be shared. What
        *is* shared -- and what makes the batch worth having -- is one policy
        resolution and one item-parameter read across the union of all pools,
        instead of one of each per user.
        """
        resolved, policy_id, policy_cons = self._resolve_policy(
            tenant, goal, tune, policy_ref)
        utility = resolved.utility
        constraints_spec = _merge_constraints(policy_cons, constraints_spec)


        mtor, items = self._mtor_for(tenant)
        scorer = UtilityScorer(mtor)

        within = (constraints_spec or {}).get("within_tags") or None
        prepared = []
        union: set[str] = set()
        for uid in user_ids:
            belief, known = self._load_belief(mtor, tenant, uid, now)
            if candidate_ids is not None:
                pool = self.store.get_items(tenant, candidate_ids,
                                            limit=self.cfg.recall_limit)
                meta = {"strategy": "explicit_ids", "scanned": len(pool)}
            else:
                pool, meta = self._recall(tenant, scorer, belief, utility, count,
                                          seed=_seed_of(uid), within_tags=within)

            prepared.append((uid, belief, known, pool, meta))
            union.update(it.id for it in pool)

        if union:
            items.preload(sorted(union))

        return [self._decide_one(tenant, uid, count, resolved, utility, policy_id,
                                 scorer, belief, known, pool, meta,
                                 constraints_spec, now, seed=None)
                for uid, belief, known, pool, meta in prepared]

    def decide(
        self,
        tenant: str,
        user_id: str,
        count: int,
        goal: str | None = None,
        tune: Mapping[str, Any] | None = None,
        constraints_spec: Mapping[str, Any] | None = None,
        candidate_ids: Sequence[str] | None = None,
        now: float = 0.0,
        seed: int | None = None,
        policy_ref: str | None = None,
    ) -> DecideResult:
        resolved, policy_id, policy_cons = self._resolve_policy(
            tenant, goal, tune, policy_ref)
        utility = resolved.utility
        constraints_spec = _merge_constraints(policy_cons, constraints_spec)


        mtor, items = self._mtor_for(tenant)
        scorer = UtilityScorer(mtor)
        belief, known = self._load_belief(mtor, tenant, user_id, now)

        if candidate_ids is not None:
            pool = self.store.get_items(tenant, candidate_ids, limit=self.cfg.recall_limit)
            meta = {"strategy": "explicit_ids", "scanned": len(pool)}
        else:
            pool, meta = self._recall(
                tenant, scorer, belief, utility, count, seed=_seed_of(user_id),
                within_tags=(constraints_spec or {}).get("within_tags") or None)

        if pool:
            items.preload([it.id for it in pool])

        return self._decide_one(tenant, user_id, count, resolved, utility, policy_id,
                                scorer, belief, known, pool, meta, constraints_spec,
                                now, seed)

    def _decide_one(self, tenant, user_id, count, resolved, utility, policy_id,
                    scorer, belief, known, pool, recall_meta, constraints_spec,
                    now, seed) -> DecideResult:
        decision_id = uuid.uuid4().hex
        timer = Timer()

        if not pool:
            dec = Decision(chosen=(), confidence="low",
                           fallback_reason="empty_candidate_pool",
                           model_version=self.model_version, policy_id=policy_id)
            self._log(tenant, decision_id, user_id, resolved.goal, policy_id, dec, now,
                      recall_meta)
            self.metrics.incr("decide", tenant=tenant, outcome="empty_pool")
            self.metrics.observe_latency("decide", timer.ms)
            return DecideResult(dec, decision_id, resolved.goal, resolved.applied,
                                resolved.notes, resolved.warnings, recall_meta)

        constraints = constraints_from(constraints_spec, k=count)

        chooser = GreedyChooser(
            seed=seed if seed is not None else _seed_of(decision_id),
            explore_pool_factor=self.cfg.explore_pool_factor)

        scored = scorer.value(belief, pool, utility)
        dec = chooser.solve(scored, pool, utility, constraints)

        confidence = dec.confidence
        reason = dec.fallback_reason
        if not known and confidence == "high":
            # A cold user got a valid set, but it rests on the prior, not on
            # evidence. Saying so is more useful than a confident-looking guess.
            confidence, reason = "medium", "cold_start_no_signals"

        dec = Decision(chosen=dec.chosen, confidence=confidence, fallback_reason=reason,
                       model_version=self.model_version, policy_id=policy_id)

        if dec.chosen:
            self.store.log_predictions(
                tenant, user_id, decision_id, self.model_version,
                {c.item_id: c.p_hat for c in dec.chosen}, now)

        self._log(tenant, decision_id, user_id, resolved.goal, policy_id, dec, now,
                  recall_meta)
        self.metrics.incr("decide", tenant=tenant, outcome=confidence)
        self.metrics.observe_latency("decide", timer.ms)
        return DecideResult(dec, decision_id, resolved.goal, resolved.applied,
                            resolved.notes, resolved.warnings, recall_meta)

    # -- decision lookup ---------------------------------------------------


    def get_decision(self, tenant: str, decision_id: str) -> dict[str, Any] | None:
        """What was served, by id.

        The point is that a caller does not have to keep this. Everything needed
        to close the loop -- the item list, each item's propensity, the policy and
        model versions -- is already stored for audit, so making the client
        maintain a parallel copy was duplicated bookkeeping with a silent failure
        mode when the copy was lost.
        """
        return self.store.get_decision(tenant, decision_id)





    # -- observe ----------------------------------------------------------

    def observe(self, tenant: str, signals: Sequence[Mapping[str, Any]],
                now: float = 0.0) -> ObserveResult:
        """Fold a batch of interactions in. Each dict needs user_id, item_id,
        outcome, ts and a signal_id; propensity/policy_id are optional but
        recorded for off-policy correction when present.

        If a signal omits ``propensity`` but carries a ``decision_id``, the
        propensity is looked up from the stored decision. This is what lets a
        caller close the loop by echoing back only the decision_id, instead of
        persisting a per-item propensity map itself -- the piece of bookkeeping
        most likely to be lost, and impossible to reconstruct once it is.

        The whole batch runs in one immediate transaction: idempotency claim,
        belief read-modify-write and item-parameter flush are atomic together,
        so a mid-batch failure leaves no partially-applied user.
        """
        accepted = duplicates = unknown = backfilled = missing_prop = 0
        with self.store.transaction() as con:
            mtor, items = self._mtor_for(tenant, con=con)

            # One catalogue read and one parameter read for the whole batch. Per
            # signal lookups turn a 500-signal batch into 1000 queries, which is
            # how an ingestion endpoint becomes the bottleneck.
            wanted = sorted({s["item_id"] for s in signals})
            catalogue = {it.id: it for it in self.store.get_items(tenant, wanted)}
            items.preload(list(catalogue))

            # Resolve propensities for referenced decisions in one batched read,
            # for the same reason: the per-signal alternative is a query storm.
            prop_by_decision = self.store.propensities_for(
                tenant, [s.get("decision_id") for s in signals], con=con)

            # Group by user so each user's belief is loaded and written once even
            # when the batch interleaves users.
            by_user: dict[str, list[Mapping[str, Any]]] = {}
            for s in signals:
                by_user.setdefault(s["user_id"], []).append(s)

            for user_id, group in by_user.items():
                belief, _ = self._load_belief(mtor, tenant, user_id, now, con=con)
                touched = False
                for s in sorted(group, key=lambda x: float(x["ts"])):
                    item = catalogue.get(s["item_id"])
                    if item is None:
                        unknown += 1
                        continue

                    propensity = s.get("propensity")
                    if propensity is None and s.get("decision_id"):
                        propensity = prop_by_decision.get(
                            s["decision_id"], {}).get(s["item_id"])
                        if propensity is not None:
                            backfilled += 1

                    sid = s.get("signal_id") or uuid.uuid4().hex
                    claimed = self.store.claim_signal(
                        tenant, sid, user_id, s["item_id"], float(s["outcome"]),
                        float(s["ts"]), propensity, s.get("policy_id"),
                        self.model_version, now, con)
                    if not claimed:
                        duplicates += 1
                        continue

                    if propensity is None:
                        # Trains the belief but is useless for off-policy work.
                        # Counted, not dropped, so the gap shows up in the response.
                        missing_prop += 1

                    # Online calibration: score the prediction we actually served
                    # against the outcome that came back. Consumed, so a re-served
                    # item cannot be counted twice.
                    p_served = self.store.take_prediction(tenant, user_id, s["item_id"], con)
                    if p_served is not None:
                        self.metrics.calibration.record(p_served, float(s["outcome"]))

                    sig = Signal(user_id=user_id, item_id=s["item_id"],
                                 outcome=float(s["outcome"]), ts=float(s["ts"]),
                                 context=s.get("context", {}),
                                 propensity=propensity)
                    belief = mtor.update(belief, sig, item)
                    accepted += 1
                    touched = True
                if touched:
                    self.store.save_belief(tenant, belief, now, con=con)
            items.flush()
        self.metrics.incr("observe_accepted", accepted, tenant=tenant)
        self.metrics.incr("observe_duplicate", duplicates, tenant=tenant)
        self.metrics.incr("observe_unknown_item", unknown, tenant=tenant)
        if missing_prop:
            self.metrics.incr("observe_missing_propensity", missing_prop, tenant=tenant)
        return ObserveResult(accepted=accepted, duplicates=duplicates,
                             unknown_items=unknown, backfilled_propensity=backfilled,
                             missing_propensity=missing_prop)



    # -- ingestion of catalogue & taxonomy --------------------------------

    def register_items(self, tenant: str, items: Sequence[Item]) -> dict[str, int]:
        tags: set[str] = set()
        for it in items:
            tags.update(it.tag_weights)
        self.store.ensure_tags(tenant, tags)
        counts = self.store.upsert_items(tenant, items)
        return {**counts, "tags": self.store.tag_space(tenant).n_dims}

    def list_items(self, tenant: str, after: str | None = None,
                   limit: int = 200) -> dict[str, Any]:
        """One page of the catalogue plus a cursor.

        Read-back for the write-only ingestion path: without it a caller cannot
        confirm what was registered, how its tags were parsed, or whether an id
        collided and overwrote something.
        """
        page = self.store.list_items(tenant, after=after, limit=limit)
        items = [{"id": it.id, "tags": dict(it.tag_weights),
                  "difficulty_prior": it.difficulty_prior, "attrs": dict(it.attrs)}
                 for it in page]
        next_after = page[-1].id if len(page) == limit else None
        return {"items": items, "count": len(items), "next_after": next_after,
                "total": self.store.item_count(tenant)}

    # -- policies (L3 escape hatch) ---------------------------------------

    def register_policy(self, tenant: str, doc: Mapping[str, Any],
                        now: float = 0.0) -> dict[str, Any]:
        """Validate and persist a tenant policy.

        Validation is the same one the shipped catalogue passes at load, run here
        so a bad policy is rejected at registration -- while someone is looking --
        rather than on the first request that names it. A negative structure
        weight, for instance, is refused here, not silently allowed to void the
        chooser's approximation guarantee.
        """
        resolved = validate_policy_doc(doc, self.catalog)   # raises PolicyError
        ref = str(doc["id"])
        self.store.save_policy(tenant, ref, dict(doc), doc.get("label"), now)
        return {"policy_ref": ref, "warnings": list(resolved.warnings)}

    def list_policies(self, tenant: str) -> list[dict[str, Any]]:
        return self.store.list_policies(tenant)

    def get_policy(self, tenant: str, policy_ref: str) -> dict[str, Any] | None:
        return self.store.get_policy(tenant, policy_ref)

    def delete_policy(self, tenant: str, policy_ref: str) -> bool:
        return self.store.delete_policy(tenant, policy_ref)


    # -- profile ----------------------------------------------------------

    def profile(self, tenant: str, user_id: str, now: float = 0.0,
                top: int = 20) -> dict[str, Any]:
        """Expose state without exposing logits.

        ``level`` is sigmoid(mu) in [0,1] and ``confidence`` falls as variance
        rises. Returning raw logits would force every caller to learn the model's
        internal scale, and would make the scale an accidental part of the public
        contract.
        """
        mtor, _ = self._mtor_for(tenant)
        belief, known = self._load_belief(mtor, tenant, user_id, now)
        space = belief.space
        if space.n_dims == 0:
            return {"user_id": user_id, "known": False, "tags": [],
                    "fallback_reason": "no_taxonomy_registered"}

        level = 1.0 / (1.0 + np.exp(-np.clip(belief.mu, -30, 30)))
        conf = 1.0 / (1.0 + belief.var)
        rows = [
            {"tag": space.tag_of[j], "level": round(float(level[j]), 4),
             "confidence": round(float(conf[j]), 4)}
            for j in range(space.n_dims)
            if space.tag_of[j] != "_latent"
        ]
        rows.sort(key=lambda r: r["level"])
        return {
            "user_id": user_id,
            "known": known,
            "model_version": self.model_version,
            "weakest": rows[:top],
            "strongest": list(reversed(rows[-top:])),
            "fallback_reason": None if known else "cold_start_no_signals",

        }

    # -- internals --------------------------------------------------------

    def _log(self, tenant: str, decision_id: str, user_id: str, goal: str,
             policy_id: str, dec: Decision, now: float,
             recall: Mapping[str, Any] | None = None) -> None:
        payload = {
            "confidence": dec.confidence,
            "fallback_reason": dec.fallback_reason,
            "recall": dict(recall or {}),
            "chosen": [
                {"item_id": c.item_id, "utility": c.utility, "p_hat": c.p_hat,
                 "propensity": c.propensity, "reasons": dict(c.reasons)}
                for c in dec.chosen
            ],
        }

        self.store.log_decision(tenant, decision_id, user_id, goal, policy_id,
                                self.model_version, dec.confidence, payload, now)


def _fit(a: np.ndarray, n: int, fill: float) -> np.ndarray:
    if a.size == n:
        return a.copy()
    out = np.full(n, fill, dtype=np.float64)
    out[:min(a.size, n)] = a[:min(a.size, n)]
    return out


def _seed_of(decision_id: str) -> int:
    return int(hashlib.sha256(decision_id.encode()).hexdigest()[:8], 16)


def _dedupe_items(items: Sequence[Item]) -> list[Item]:
    """First occurrence wins, order preserved.

    The chooser keys candidates by id, so a pool holding one id twice lets the
    same item be selected twice in one slate. Order is preserved because the
    objective slice comes first and is the one worth keeping.
    """
    seen: set[str] = set()
    out: list[Item] = []
    for it in items:
        if it.id in seen:
            continue
        seen.add(it.id)
        out.append(it)
    return out

