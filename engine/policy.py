"""Policy loading -- the translation layer between the external surface and the
kernel.

Outside, a caller says ``goal="practice_weak"`` and maybe ``tune={"difficulty":
"hard"}``. Inside, the engine only understands ``Utility(rho, gamma, V, Phi)``.
This module is the only place those two vocabularies meet, and that is the point:
if a customer ever has to know that ``rho.target`` exists in order to get what
they want, the translation layer has failed and the abstraction has leaked
outward.

The mapping table itself lives in ``contracts/goals.yaml`` rather than here, so
that adding a goal is a data change reviewable by someone who does not read
Python.

Validation
----------
The rules in ``goals.yaml`` are declarative strings, deliberately *not* an
executable DSL -- a second expression language would be a second parser to keep
correct. Each rule id instead has a named implementation below, and load time
asserts the two sets match exactly. A rule added to the YAML with no
implementation is therefore a load failure rather than a rule that silently
never fires, which is the failure mode that matters.

Rules fire at load time, not at request time. A misconfigured policy should fail
when it is installed, while someone is looking at it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from contracts.core import (
    Constraints,
    Quota,
    RewardSpec,
    StructureSpec,
    Utility,
    ValueSpec,
)
from engine.predicates import PATH_SYNTAX, PredicateError, compile_all, is_valid_path
from engine.transition import TRANSITIONS, TransitionError, get_transition


__all__ = ["PolicyError", "GoalCatalog", "load_catalog", "constraints_from",
           "validate_policy_doc"]

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "contracts" / "goals.yaml"

# The kernel's executable vocabulary. Kept here because this is the layer that
# decides whether a document can be installed; scorer/chooser raise on an unknown
# kind, and reaching them means the load step let something through.
REWARD_KINDS = frozenset({"peak", "increasing", "decreasing", "threshold", "constant"})
VALUE_KINDS = frozenset({"mastery_sum", "neg_entropy", "info_gain"})
DIM_WEIGHTS = frozenset({"uniform", "low_mu", "high_mu", "high_var", "target"})
STRUCTURE_KINDS = frozenset({"diversify", "concentrate", "balanced"})
SIMILARITIES = frozenset({"tag_jaccard", "tag_cosine"})
PROVENANCE_STATUSES = frozenset({"uncalibrated", "calibrating", "calibrated"})


class PolicyError(ValueError):
    """Raised for a policy that cannot be loaded or violates an error-severity rule."""


def _check_enum(where: str, value: Any, allowed: frozenset[str] | set[str]) -> None:
    if value not in allowed:
        raise PolicyError(f"{where}={value!r} is not supported{_suggest(value, allowed)}")



def _suggest(value: Any, options: Any) -> str:
    """`" (did you mean 'moderate'?)"` or empty.

    A rejection that lists 40 valid values is technically complete and
    practically useless; the near-miss is what the caller actually needs, and
    typing 'medium' for 'moderate' is the mistake everyone makes once.
    """
    opts = sorted(str(o) for o in options)
    if not isinstance(value, str) or not opts:
        return f"; choices: {opts}" if opts else ""
    close = difflib.get_close_matches(value, opts, n=2, cutoff=0.5)
    hint = f" (did you mean {' or '.join(repr(c) for c in close)}?)" if close else ""
    return f"{hint}; choices: {opts}"



# ---------------------------------------------------------------------------
# validation rules -- one function per rule id in goals.yaml
# ---------------------------------------------------------------------------

_LEARNING_GOALS = ("practice_weak", "challenge", "review")


def _r_learning_goals_need_peak(goal: str, u: Utility) -> str | None:
    if goal in _LEARNING_GOALS and u.rho.kind != "peak":
        return f"goal={goal!r} has rho.kind={u.rho.kind!r}; learning goals require 'peak'"
    return None


def _r_amplification_needs_explore_floor(goal: str, u: Utility) -> str | None:
    concentrating = u.structure is not None and u.structure.kind == "concentrate"
    if u.gamma == 0.0 and concentrating and u.explore_floor < 0.05:
        return (f"goal={goal!r} is pure exploitation with concentrate structure but "
                f"explore_floor={u.explore_floor}; needs >= 0.05")
    return None


def _r_lookahead_needs_transition_model(goal: str, u: Utility) -> str | None:
    if u.gamma > 0.0:
        if not u.transition_model:
            return f"goal={goal!r} has gamma={u.gamma} > 0 but no transition_model"
        # The model must exist AND its mean-shift hypothesis is a domain claim,
        # so a typo here is a silent behaviour change, not a no-op. Resolve it
        # at load time to fail loudly while someone is looking.
        try:
            get_transition(u.transition_model)
        except TransitionError as e:
            return f"goal={goal!r} transition_model invalid: {e}"
    return None



def _r_target_dim_weight_needs_target(goal: str, u: Utility) -> str | None:
    if u.value is not None and u.value.dim_weight == "target" and not u.value.target:
        return f"goal={goal!r} uses dim_weight='target' without value.target"
    return None


def _r_peak_target_in_range(goal: str, u: Utility) -> str | None:
    if u.rho.kind == "peak":
        t = u.rho.target
        if t is None or not (0.30 <= t <= 0.95):
            return f"goal={goal!r} has rho.target={t}; expected within [0.30, 0.95]"
    return None


def _r_structure_weight_nonnegative(goal: str, u: Utility) -> str | None:
    if u.structure is not None and u.structure.weight < 0.0:
        # A negative weight would flip Phi's sign, destroying submodularity and
        # with it the (1 - 1/e) guarantee the chooser advertises.
        return f"goal={goal!r} has structure.weight={u.structure.weight} < 0"
    return None


_RULES: Mapping[str, Callable[[str, Utility], str | None]] = {
    "learning_goals_need_peak": _r_learning_goals_need_peak,
    "amplification_needs_explore_floor": _r_amplification_needs_explore_floor,
    "lookahead_needs_transition_model": _r_lookahead_needs_transition_model,
    "target_dim_weight_needs_target": _r_target_dim_weight_needs_target,
    "peak_target_in_range": _r_peak_target_in_range,
    "structure_weight_nonnegative": _r_structure_weight_nonnegative,
}


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """A Utility plus the audit trail of how it was produced.

    ``notes`` records tune keys that were accepted but had no effect (asking for
    'hard' on a goal with no success target, for instance). Silently dropping
    them would let a customer believe a knob is doing something when it is not.
    """

    goal: str
    utility: Utility
    applied: Mapping[str, Any]
    notes: tuple[str, ...]
    warnings: tuple[str, ...]


class GoalCatalog:
    def __init__(self, doc: Mapping[str, Any]) -> None:
        self.doc = doc
        self.goals: Mapping[str, Any] = doc.get("goals") or {}
        self.tune_maps: Mapping[str, Any] = doc.get("tune_maps") or {}
        self.default_goal: str = doc.get("default_goal") or ""
        self.provenance: Mapping[str, Any] = doc.get("provenance") or {}

        if not self.goals:
            raise PolicyError("goals.yaml declares no goals")
        if self.default_goal not in self.goals:
            raise PolicyError(f"default_goal {self.default_goal!r} is not a declared goal")
        if "status" not in self.provenance:
            raise PolicyError("goals.yaml declares no provenance.status")
        _check_enum("provenance.status", self.provenance["status"], PROVENANCE_STATUSES)

        declared = {r["id"] for r in doc.get("validation") or []}
        missing = declared - set(_RULES)
        if missing:
            raise PolicyError(f"validation rules with no implementation: {sorted(missing)}")
        unused = set(_RULES) - declared
        if unused:
            raise PolicyError(f"implemented rules absent from goals.yaml: {sorted(unused)}")
        self.severity: Mapping[str, str] = {
            r["id"]: r.get("severity", "error") for r in doc.get("validation") or []
        }

        # Every declared goal must be valid on its own. Shipping a catalogue
        # whose entries only pass under particular tune values would mean the
        # zero-config path is the broken one.
        for name in self.goals:
            self.resolve(name)

        # A goal must not restate its parameters as tune adjectives. Applying
        # such a block would overwrite the utility it sits next to (a goal
        # declaring structure.weight 1.20 alongside focus='broad' would silently
        # run at 0.80), leaving one of the two numbers a lie.
        redundant = [n for n, g in self.goals.items() if g.get("defaults")]
        if redundant:
            raise PolicyError(
                f"goals declare a 'defaults' block, which duplicates their utility "
                f"block: {sorted(redundant)}. Adjectives are derived via describe()."
            )


    @property
    def calibrated(self) -> bool:
        return self.provenance.get("status") == "calibrated"

    # -- goal -> Utility --------------------------------------------------

    def _base_utility(self, goal: str) -> Utility:
        return self._utility_from_spec(self.goals[goal].get("utility") or {})

    @staticmethod
    def _utility_from_spec(spec: Mapping[str, Any]) -> Utility:
        """Build a Utility, rejecting anything the kernel cannot execute.

        Enum values are checked here, at load/registration time, rather than being
        passed through to be discovered by ``scorer``/``chooser`` mid-request. A
        misspelt ``structure.kind`` used to install cleanly and then raise on every
        request that named the policy -- which the service caught as a soft
        degradation, so the policy looked installed and silently returned fallbacks
        forever. Rejecting here is the whole point of having a load step.
        """
        rho_raw = spec.get("rho") or {}
        rho_kind = rho_raw.get("kind", "peak")
        _check_enum("rho.kind", rho_kind, REWARD_KINDS)
        if rho_kind in ("peak", "threshold") and rho_raw.get("target") is None:
            raise PolicyError(f"rho.kind={rho_kind!r} requires 'target'")
        target = rho_raw.get("target")
        if target is not None:
            target = float(target)
            if not 0.0 <= target <= 1.0:
                # A success probability outside [0, 1] is not a tuning choice, it is
                # a unit mistake. scorer clamps and carries on, so left unchecked it
                # runs forever at the clamp instead of saying anything.
                raise PolicyError(f"rho.target must be in [0, 1], got {target}")
        width = float(rho_raw.get("width", 0.15))
        if not 0.0 < width <= 1.0:
            # width 0 is a zero-tolerance band (scorer only avoids the division via
            # a 1e-6 floor); negative is meaningless.
            raise PolicyError(f"rho.width must be in (0, 1], got {width}")
        rho = RewardSpec(
            kind=rho_kind,
            target=target,
            width=width,
            value_attr=rho_raw.get("value_attr"),
        )

        if rho.value_attr and not is_valid_path(rho.value_attr):
            raise PolicyError(
                f"rho.value_attr {rho.value_attr!r} is not an addressable path "
                f"({PATH_SYNTAX})")

        val_raw = spec.get("value")
        value = None
        if val_raw:
            if "kind" not in val_raw:
                raise PolicyError("value block requires 'kind'")
            _check_enum("value.kind", val_raw["kind"], VALUE_KINDS)
            dim_weight = val_raw.get("dim_weight", "uniform")
            _check_enum("value.dim_weight", dim_weight, DIM_WEIGHTS)
            if dim_weight == "target" and not val_raw.get("target"):
                raise PolicyError("value.dim_weight='target' requires 'target'")
            value = ValueSpec(
                kind=val_raw["kind"],
                dim_weight=dim_weight,
                target=val_raw.get("target"),
            )

        st_raw = spec.get("structure")
        structure = None
        if st_raw:
            if "kind" not in st_raw:
                raise PolicyError("structure block requires 'kind'")
            _check_enum("structure.kind", st_raw["kind"], STRUCTURE_KINDS)
            similarity = st_raw.get("similarity", "tag_jaccard")
            _check_enum("structure.similarity", similarity, SIMILARITIES)
            if st_raw["kind"] == "balanced" and st_raw.get("target_entropy") is None:
                # The chooser falls back to target 0.0, which drives tag entropy to
                # zero -- i.e. it silently behaves like `concentrate`. "Balanced
                # toward nothing in particular" is not a thing anyone means.
                raise PolicyError("structure.kind='balanced' requires 'target_entropy'")
            structure = StructureSpec(

                kind=st_raw["kind"],
                weight=float(st_raw.get("weight", 0.0)),
                similarity=similarity,
                target_entropy=st_raw.get("target_entropy"),
            )

        gamma = float(spec.get("gamma", 0.0))
        if not 0.0 <= gamma <= 1.0:
            # gamma trades immediate reward against expected state change; outside
            # [0, 1] it either inverts the trade or lets lookahead swamp rho
            # entirely, and nothing downstream would report either.
            raise PolicyError(f"gamma must be in [0, 1], got {gamma}")
        transition_model = spec.get("transition_model")

        if transition_model is not None:
            _check_enum("transition_model", transition_model, set(TRANSITIONS))
        if gamma > 0.0 and not transition_model:
            # get_transition() refuses a default because the choice is a domain
            # claim; surfacing that at load beats surfacing it per request.
            raise PolicyError(
                "gamma > 0 requires an explicit transition_model "
                f"(one of {sorted(TRANSITIONS)})")

        explore_floor = float(spec.get("explore_floor", 0.0))
        if not 0.0 <= explore_floor < 1.0:
            raise PolicyError(f"explore_floor must be in [0, 1), got {explore_floor}")

        return Utility(
            rho=rho,
            gamma=gamma,
            value=value,
            structure=structure,
            explore_floor=explore_floor,
            transition_model=transition_model,
        )



    def resolve(
        self,
        goal: str | None = None,
        tune: Mapping[str, Any] | None = None,
        strict_tune: bool = True,
    ) -> Resolved:
        """L0/L1/L2 in one call: no arguments lands on the default goal."""
        name = goal or self.default_goal
        if name not in self.goals:
            raise PolicyError(f"unknown goal {name!r}{_suggest(name, self.goals)}")

        u = self._base_utility(name)
        applied: dict[str, Any] = {}
        notes: list[str] = []
        for key, raw in (dict(tune) if tune else {}).items():
            if key not in self.tune_maps:
                if strict_tune:
                    raise PolicyError(
                        f"unknown tune key {key!r}{_suggest(key, self.tune_maps)}"
                    )
                notes.append(f"tune key {key!r} ignored (unknown)")
                continue
            u, effect, note = self._apply_tune(u, key, raw)
            if note:
                notes.append(note)
            if effect is not None:
                applied[key] = effect


        warnings = self._validate(name, u)
        return Resolved(goal=name, utility=u, applied=applied,
                        notes=tuple(notes), warnings=tuple(warnings))

    # -- L3: a caller-supplied policy document ----------------------------

    def resolve_doc(
        self,
        doc: Mapping[str, Any],
        tune: Mapping[str, Any] | None = None,
    ) -> tuple[Resolved, dict[str, Any]]:
        """Resolve a full policy document (``policy.schema.json``) to a Utility.

        This is the L3 escape hatch actually wired up. It runs the *same*
        validation rules as the shipped catalogue, so a tenant cannot install a
        policy that the engine would have rejected in ``goals.yaml`` -- notably a
        negative structure weight, which would quietly void the (1-1/e) guarantee.

        ``extends`` overlays the document on a built-in goal, which is how most
        custom needs should be expressed: inheriting means a later change to the
        base goal is picked up instead of being frozen into a copy.

        Returns the Resolved plus the document's constraints block, because at L3
        the feasible set is part of the policy rather than of the request.
        """
        if not isinstance(doc, Mapping):
            raise PolicyError("policy document must be an object")
        pid = str(doc.get("id") or "").strip()
        if not pid:
            raise PolicyError("policy document requires an 'id'")

        believe = doc.get("believe")
        if believe not in (None, "", "mtor"):
            # Declared in the schema, not implemented. Saying so beats accepting
            # the field and running a different model than the caller asked for.
            raise PolicyError(
                f"believe={believe!r} is not available; only 'mtor' is implemented")

        base_spec: Mapping[str, Any] = {}
        extends = doc.get("extends")
        if extends:
            if extends not in self.goals:
                raise PolicyError(
                    f"extends {extends!r} is not a known goal{_suggest(extends, self.goals)}")
            base_spec = self.goals[extends].get("utility") or {}
        spec = _merge(base_spec, doc.get("utility") or {})
        if not spec:
            raise PolicyError("policy document declares neither 'utility' nor 'extends'")

        u = self._utility_from_spec(spec)
        applied: dict[str, Any] = {}
        notes: list[str] = []
        for key, raw in (dict(tune) if tune else {}).items():
            if key not in self.tune_maps:
                raise PolicyError(f"unknown tune key {key!r}{_suggest(key, self.tune_maps)}")
            u, effect, note = self._apply_tune(u, key, raw)
            if note:
                notes.append(note)
            if effect is not None:
                applied[key] = effect

        warnings = list(self._validate(pid, u))
        status = (doc.get("provenance") or {}).get("status")
        if status is None:
            # The same bar the shipped catalogue is held to: a number with no
            # stated origin is the failure mode this project exists to avoid.
            warnings.append("[provenance] policy declares no provenance.status; "
                            "treat its constants as uncalibrated")
        else:
            # A status outside the enum used to be read as "not calibrated",
            # so 'calibrated_probably' silently downgraded the whole document
            # while looking like an assertion of the opposite.
            _check_enum("provenance.status", status, PROVENANCE_STATUSES)

        cons = dict(doc.get("constraints") or {})
        if cons.get("exclude_seen_within_days") is not None:
            raise PolicyError(
                "constraints.exclude_seen_within_days appears in policy.schema.json but "
                "is not implemented; exclude explicitly via exclude.item_ids for now")

        resolved = Resolved(goal=pid, utility=u, applied=applied,
                            notes=tuple(notes), warnings=tuple(warnings))
        return resolved, cons


    def utility(self, goal: str | None = None, tune: Mapping[str, Any] | None = None) -> Utility:
        return self.resolve(goal, tune).utility

    # -- goal -> adjectives (derived, never stored) ------------------------

    def describe(self, goal: str) -> dict[str, Any]:
        """What a goal looks like in the external vocabulary.

        Derived by reverse lookup against ``tune_maps``, so it cannot drift out
        of step with the utility it describes. The adjective returned is the
        nearest bucket, which is honest: goals are tuned more finely than the
        four words a caller gets to say, and the description says "about hard",
        not "exactly hard".
        """
        u = self._base_utility(goal) if goal in self.goals else None
        if u is None:
            raise PolicyError(f"unknown goal {goal!r}")

        out: dict[str, Any] = {
            "label": self.goals[goal].get("label", goal),
            "intent": self.goals[goal].get("intent", ""),
            "freshness": round(u.explore_floor, 3),
        }
        if u.rho.kind in ("peak", "threshold") and u.rho.target is not None:
            out["difficulty"] = self._nearest("difficulty", u.rho.target)
            out["stakes"] = self._nearest("stakes", u.rho.width)
        if u.structure is not None:
            out["focus"] = next(
                (k for k, v in self.tune_maps["focus"].items()
                 if not k.startswith("_") and v.get("kind") == u.structure.kind),
                None,
            )
        return out

    def _nearest(self, key: str, value: float) -> str | None:
        opts = [(k, v) for k, v in self.tune_maps[key].items()
                if not k.startswith("_") and isinstance(v, (int, float))]
        if not opts:
            return None
        return min(opts, key=lambda kv: abs(float(kv[1]) - value))[0]


    # -- tune -> parameter ------------------------------------------------

    def _lookup(self, key: str, raw: Any) -> Any:
        table = self.tune_maps[key]
        if isinstance(raw, str):
            if raw not in table or raw.startswith("_"):
                choices = [k for k in table if not k.startswith("_")]
                raise PolicyError(
                    f"tune {key}={raw!r} not recognised{_suggest(raw, choices)}")
            return table[raw]
        return raw          # numeric pass-through, e.g. freshness=0.25

    def _apply_tune(self, u: Utility, key: str, raw: Any) -> tuple[Utility, Any, str | None]:
        target_field = self.tune_maps[key].get("_maps_to")
        v = self._lookup(key, raw)

        if target_field == "rho.target":
            if u.rho.kind not in ("peak", "threshold"):
                # increasing / decreasing / constant have no success target, so
                # the knob is genuinely inert here rather than subtly applied.
                return u, None, f"tune {key}={raw!r} has no effect on rho.kind={u.rho.kind!r}"
            return replace(u, rho=replace(u.rho, target=float(v))), float(v), None

        if target_field == "rho.width":
            return replace(u, rho=replace(u.rho, width=float(v))), float(v), None

        if target_field == "explore_floor":
            f = float(v)
            # Same bound as the utility block: 1.0 would spend every slot on
            # exploration, leaving no exploitation at all. Two different bounds for
            # one field would mean `tune` could reach a state the policy document
            # is forbidden to declare.
            if not (0.0 <= f < 1.0):
                raise PolicyError(f"tune {key}={raw!r} outside [0, 1)")
            return replace(u, explore_floor=f), f, None


        if target_field == "structure":
            if not isinstance(v, Mapping):
                raise PolicyError(f"tune {key}={raw!r} must map to a structure block")
            base = u.structure
            st = StructureSpec(
                kind=v.get("kind", base.kind if base else "diversify"),
                weight=float(v.get("weight", base.weight if base else 0.0)),
                similarity=v.get("similarity", base.similarity if base else "tag_jaccard"),
                target_entropy=v.get("target_entropy",
                                     base.target_entropy if base else None),
            )
            # Re-checked because a tune block rebuilds the spec rather than
            # inheriting the validated one: a typo in goals.yaml's tune_maps would
            # otherwise slip past load and only raise inside the chooser.
            _check_enum(f"tune {key}.kind", st.kind, STRUCTURE_KINDS)
            _check_enum(f"tune {key}.similarity", st.similarity, SIMILARITIES)
            if st.kind == "balanced" and st.target_entropy is None:
                raise PolicyError(
                    f"tune {key}={raw!r} selects 'balanced' without target_entropy")
            return replace(u, structure=st), {"kind": st.kind, "weight": st.weight}, None



        raise PolicyError(f"tune key {key!r} declares unsupported _maps_to={target_field!r}")

    # -- rules ------------------------------------------------------------

    def _validate(self, goal: str, u: Utility) -> list[str]:
        warnings: list[str] = []
        for rule_id, fn in _RULES.items():
            msg = fn(goal, u)
            if msg is None:
                continue
            if self.severity.get(rule_id, "error") == "error":
                raise PolicyError(f"[{rule_id}] {msg}")
            warnings.append(f"[{rule_id}] {msg}")
        return warnings


def load_catalog(path: str | Path | None = None) -> GoalCatalog:
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return GoalCatalog(yaml.safe_load(fh))


def _merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``over`` onto ``base`` (for policy ``extends``).

    Nested maps merge key-wise; a scalar or list in ``over`` replaces the base.
    So a policy extending ``practice_weak`` can override just ``rho.target``
    without having to restate the whole rho block.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_policy_doc(doc: Mapping[str, Any], catalog: GoalCatalog | None = None) -> Resolved:
    """Load-time check for an L3 policy document, standalone.

    Used by the registration endpoint so a policy is rejected when it is
    installed -- while someone is looking at it -- rather than on the first
    request that references it.
    """
    cat = catalog or load_catalog()
    resolved, _ = cat.resolve_doc(doc)
    return resolved



# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def constraints_from(spec: Mapping[str, Any] | None, k: int) -> Constraints:
    """Build a feasible set from the wire representation.

    Predicates are compiled here so a malformed expression is a load/registration
    error. Leaving compilation to the chooser meant it happened per request, where
    the service catches it as a soft degradation -- so a policy with a bad
    expression installed cleanly and then returned fallbacks forever.
    """
    spec = spec or {}
    quotas = tuple(
        Quota(group_by=q["group_by"], counts={str(a): int(b) for a, b in q["counts"].items()})
        for q in spec.get("quotas") or []
    )
    total = sum(sum(q.counts.values()) for q in quotas)
    if total > k:
        raise PolicyError(f"quotas require {total} slots but k={k}")
    max_per_tag = spec.get("max_per_tag")
    if max_per_tag is not None:
        max_per_tag = int(max_per_tag)
        if max_per_tag < 1:
            # 0 or negative makes every tagged item inadmissible, which reads as
            # "return nothing" -- almost never the intent, and silent if allowed.
            raise PolicyError(f"max_per_tag must be >= 1, got {max_per_tag}")
    predicates = tuple(spec.get("predicates") or [])
    try:
        compile_all(predicates)
    except PredicateError as exc:
        raise PolicyError(f"invalid constraint expression: {exc}") from exc
    return Constraints(
        k=int(k),
        predicates=predicates,
        quotas=quotas,
        max_per_tag=max_per_tag,
        exclude_item_ids=frozenset(spec.get("exclude_item_ids") or ()),
        within_tags=frozenset(spec.get("within_tags") or ()),
    )


