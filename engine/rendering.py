"""Presentation -- machine-readable reasons into a sentence a person can read.

Kept out of the engine on purpose. The kernel has four nouns (user, item, tag,
signal) and no vocabulary to render into; a sentence about "questions" or
"products" is domain prose and would break the one rule the whole design rests
on. It is also kept out of ``api.py`` so it is reachable without a server, and so
templates are data rather than a Python literal a tenant has to fork the service
to change.

Two things are rendered here:

``why``   why *this* item won, chosen by which term of the utility dominated
``hint``  what to do about a degraded result, keyed by ``fallback_reason``

Both come from ``contracts/why.yaml``. Missing keys fall back to the default
locale rather than raising: a partial translation should produce a mixed-language
response, never a 500 in the middle of a served decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["Renderer", "load_renderer", "DEFAULT_PATH"]

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "contracts" / "why.yaml"


@dataclass(frozen=True)
class Renderer:
    locales: Mapping[str, Mapping[str, Any]]
    default_locale: str = "zh"

    # -- locale resolution -------------------------------------------------

    def available(self) -> list[str]:
        return sorted(self.locales)

    def _pick(self, locale: str | None) -> str:
        """Resolve a requested locale to one we have.

        Accepts ``zh-CN`` for ``zh`` because that is what browsers and most HTTP
        clients send; refusing it would make the field look broken for the most
        common way of asking.
        """
        if not locale:
            return self.default_locale
        low = locale.strip().lower()
        if low in self.locales:
            return low
        head = low.split("-")[0].split("_")[0]
        if head in self.locales:
            return head
        return self.default_locale

    def _table(self, locale: str | None, section: str) -> dict[str, str]:
        chosen = self._pick(locale)
        base = dict((self.locales.get(self.default_locale) or {}).get(section) or {})
        base.update((self.locales.get(chosen) or {}).get(section) or {})
        return base

    # -- rendering ---------------------------------------------------------

    def why(self, reasons: Mapping[str, float], p_hat: float,
            locale: str | None = None) -> str:
        """One sentence, selected by which contribution actually decided the pick.

        Derived from the reason the item won rather than being a fixed string --
        a constant sentence would be indistinguishable from having no explanation.
        """
        tpl = self._table(locale, "why")
        pct = int(round(p_hat * 100))
        if reasons.get("explore", 0.0) == 1.0:
            key = "explore"
        else:
            st = reasons.get("structure", 0.0)
            key = "structure_neg" if st < -1e-9 else "structure_pos" if st > 1e-9 else "fit"
        text = tpl.get(key) or tpl.get("fit") or "{pct}%"
        try:
            return text.format(pct=pct).strip()
        except Exception:
            # A tenant template with any stray placeholder -- ``{pct.foo}``
            # (AttributeError), ``{pct[0]}`` (TypeError), ``{pct:zz}`` (ValueError)
            # -- must degrade to the raw template, never 5xx a decision that has
            # already been made. Catching only KeyError/IndexError left the rest
            # to propagate.
            return text.strip()


    def hint(self, fallback_reason: str | None, locale: str | None = None) -> str | None:
        """What to do next, or None when there is nothing to say.

        Returning None for an unknown reason is deliberate: inventing generic
        advice for a state we have not thought about is worse than silence.
        """
        if not fallback_reason:
            return None
        text = self._table(locale, "hints").get(fallback_reason)
        return text.strip() if text else None


def load_renderer(path: str | Path | None = None) -> Renderer:
    p = Path(path) if path else DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    locales = doc.get("locales") or {}
    if not locales:
        raise ValueError(f"{p} declares no locales")
    default = doc.get("default_locale") or sorted(locales)[0]
    if default not in locales:
        raise ValueError(f"default_locale {default!r} is not among {sorted(locales)}")
    return Renderer(locales=locales, default_locale=default)
