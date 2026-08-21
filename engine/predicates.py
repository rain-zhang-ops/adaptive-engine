"""Safe predicate evaluation for customer-supplied filter expressions.

Constraints arrive as strings in a policy file, i.e. as untrusted input. Using
``eval`` here would be a straight code-execution hole, so this module implements
a deliberately tiny grammar instead. Anything outside the grammar is rejected at
load time rather than at request time.

Grammar
-------
    expr    := path OP literal
    path    := ident ( '.' ident )*          e.g. attrs.visibility, id
    OP      := == | != | < | <= | > | >= | in | not in
    literal := number | 'string' | "string" | true | false | null
             | [ literal, literal, ... ]

No boolean connectives, no arithmetic, no function calls, no attribute access on
arbitrary objects. Multiple conditions are expressed as multiple predicates,
which are ANDed by the caller -- that covers the practical cases without opening
a parser surface worth attacking.

Two limits exist so that "outside the grammar" also covers resource exhaustion
rather than only syntax: an expression length cap and a literal nesting cap. A
recursive-descent literal parser without a depth bound is a stack overflow that a
policy author reaches by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Sequence

__all__ = ["Predicate", "PredicateError", "compile_predicate", "compile_all",
           "is_valid_path", "PATH_SYNTAX"]

_OPS = ("not in", "in", "==", "!=", "<=", ">=", "<", ">")
# Segments may not begin with an underscore. Attribute traversal below is
# whitelisted to declared dataclass fields, but keeping dunder names
# unrepresentable in the grammar itself means a future traversal target cannot
# reintroduce ``__class__.__init__.__globals__`` by relaxing one check.
_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*$")
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

PATH_SYNTAX = "letter followed by letters, digits or underscores, dot-separated"

MAX_EXPR_LEN = 512
MAX_LITERAL_DEPTH = 4


class PredicateError(ValueError):
    pass


def is_valid_path(path: str) -> bool:
    """Whether ``path`` is addressable by this grammar.

    Exported so callers that *build* expressions (the HTTP layer turns
    ``exclude.attrs`` into predicates) can reject a bad key while they still have
    request context, instead of emitting a string that fails to compile later.
    """
    return bool(_PATH_RE.match(path))


def _parse_literal(text: str, depth: int = 0) -> Any:
    if depth > MAX_LITERAL_DEPTH:
        raise PredicateError(
            f"literal nested deeper than {MAX_LITERAL_DEPTH} levels: {text!r}")
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        return [_parse_literal(p, depth + 1) for p in _split_top(inner)]
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    if _NUM_RE.match(t):
        return float(t) if "." in t else int(t)
    raise PredicateError(f"unsupported literal: {text!r}")


def _split_top(text: str) -> list[str]:
    parts, buf, quote = [], [], None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class Predicate:
    path: tuple[str, ...]
    op: str
    value: Any
    source: str

    def resolve(self, obj: Any) -> Any:
        cur: Any = obj
        for part in self.path:
            if isinstance(cur, Mapping):
                cur = cur.get(part)
            elif is_dataclass(cur) and not isinstance(cur, type):
                # Declared fields only. A bare ``getattr`` here would let a path
                # walk methods and dunders -- reaching module globals via
                # ``__class__.__init__.__globals__`` -- and would fire any
                # property getter on the way, which is an execution surface even
                # though the grammar only ever compares the endpoint.
                if part not in {f.name for f in fields(cur)}:
                    return None
                cur = getattr(cur, part)
            else:
                return None
            if cur is None:
                return None
        return cur


    def holds(self, obj: Any) -> bool:
        left = self.resolve(obj)
        op, right = self.op, self.value

        if op == "in":
            return left in right if isinstance(right, (list, tuple, str)) else False
        if op == "not in":
            return left not in right if isinstance(right, (list, tuple, str)) else True
        if op == "==":
            return left == right
        if op == "!=":
            # A missing attribute satisfies "!=". Otherwise a policy such as
            # "visibility != 'embargoed'" would silently drop every item that
            # simply has no visibility attribute -- surprising, and it turns an
            # optional field into a required one.
            return left != right
        if left is None:
            return False
        try:
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
        except TypeError:
            return False
        raise PredicateError(f"unknown operator {op!r}")


def compile_predicate(expr: str) -> Predicate:
    text = expr.strip()
    if len(text) > MAX_EXPR_LEN:
        raise PredicateError(
            f"expression longer than {MAX_EXPR_LEN} chars: {text[:40]!r}...")
    # Operator search must ignore anything inside quotes, otherwise a legitimate
    # value such as ``title == 'plug in here'`` splits on the ``in`` in the
    # string and rejects a valid predicate. Scan the unquoted regions only.
    for op in _OPS:                       # longest-first, so "not in" wins over "in"
        marker = f" {op} "
        pos = _find_unquoted(text, marker)
        if pos < 0:
            continue
        raw_path = text[:pos].strip()
        raw_val = text[pos + len(marker):].strip()
        if not _PATH_RE.match(raw_path):
            raise PredicateError(f"invalid path {raw_path!r} in {expr!r}")
        return Predicate(tuple(raw_path.split(".")), op, _parse_literal(raw_val), expr)
    raise PredicateError(f"no supported operator found in {expr!r}")


def _find_unquoted(text: str, marker: str) -> int:
    """Index of ``marker`` in ``text``, skipping over quoted spans."""
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:

        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        if text.startswith(marker, i):
            return i
        i += 1
    return -1



def compile_all(exprs: Sequence[str]) -> list[Predicate]:
    return [compile_predicate(e) for e in exprs]
