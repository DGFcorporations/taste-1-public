"""core — the standards engine (taste-1 control: A1 atomic micro-layers,
H1 direct extraction, B1 cosine retrieval).

Everything that mutates taste state goes through here. The state machine is
the law:

    provisional → active → superseded|retired
       (owner confirm)      (correction wins)   (never deleted — NEVER 5)

Agents propose; only owner confirmation activates. Retired means "stopped
applying" — the record stays forever.
"""

from __future__ import annotations

from .embed import cosine, embed
from .models import (
    LayerState,
    Standard,
)
from .store import Store

TOP_N = 12          # B1: cosine top-N retrieval window
CONFIDENCE_FLOOR = 0.3   # B7-ish quality floor baked into retrieval anyway


def locally_approved(standard: Standard) -> bool:
    """A portable ``state: active`` is not local owner approval.

    Import stamps ``variant_meta.local_approved = False``. Rows this
    engine activated itself omit the flag, so older confirmed corpora
    keep applying. Retrieval and packets read only the effective set.
    """
    return standard.variant_meta.get("local_approved") is not False


class StandardsEngine:
    def __init__(self, store: Store):
        self.store = store

    # ---------------------------------------------------------- write path
    def propose(self, standard: Standard) -> Standard:
        """New layers ALWAYS arrive provisional — propose-never-apply."""
        standard.state = LayerState.PROVISIONAL
        vec = embed([standard.directive + " " + standard.rationale])[0]
        self.store.put_standard(standard, vec)
        return standard

    def confirm(self, sid: str, rationale: str | None = None) -> bool:
        """Owner confirms a comprehension restatement → layer activates.
        May supply/replace the rationale at confirmation time."""
        s = self.store.get_standard(sid)
        if not s:
            return False
        if rationale is not None:
            s.rationale = rationale
            self.store.set_rationale(sid, rationale)
        if not s.rationale.strip():
            raise ValueError(f"{sid}: cannot activate without rationale")
        meta = dict(s.variant_meta)
        meta["local_approved"] = True
        self.store.set_variant_meta(sid, meta)
        self.store.set_state(sid, LayerState.ACTIVE)
        return True

    def supersede(self, old_id: str, new: Standard) -> Standard:
        """Correction wins: old layer marked superseded (kept forever),
        new layer links back via supersedes."""
        new.supersedes = [old_id]
        new.state = LayerState.PROVISIONAL
        self.propose(new)
        self.store.set_state(old_id, LayerState.SUPERSEDED)
        return new

    def retire(self, sid: str) -> bool:
        s = self.store.get_standard(sid)
        if not s:
            return False
        self.store.set_state(sid, LayerState.RETIRED)
        return True

    def reinforce(self, sid: str, delta: float):
        """Confirmed restatements boost; contradictions decay (H5-lite —
        full confidence math stays simple in the control)."""
        s = self.store.get_standard(sid)
        if s:
            self.store.set_confidence(sid, min(1.0, max(0.0, s.confidence + delta)))

    # ---------------------------------------------------------- read path
    def retrieve(self, query: str, profile_id: str = "owner",
                 category: str | None = None, top_n: int = TOP_N
                 ) -> list[tuple[Standard, float]]:
        """B1: cosine top-N over active, above-floor standards."""
        from .models import Category
        cat = Category(category) if category else None
        rows = self.store.list_standards(
            profile_id, state=LayerState.ACTIVE, category=cat,
            with_vectors=True)
        candidates = [(s, v) for s, v in rows
                      if v is not None and s.confidence >= CONFIDENCE_FLOOR
                      and locally_approved(s)]
        if not candidates:
            return []
        qv = embed([query])[0]
        scored = [(s, cosine(qv, v)) for s, v in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]

    def active(self, profile_id: str = "owner") -> list[Standard]:
        return [s for s in self.store.list_standards(
            profile_id, state=LayerState.ACTIVE) if locally_approved(s)]

    def coverage(self, profile_id: str, task_query: str) -> dict:
        """What taste already resolves for this task + where the gaps are.
        Feeds the adaptive interview (E1)."""
        hits = self.retrieve(task_query, profile_id)
        by_cat: dict[str, int] = {}
        for s, sim in hits:
            if sim > 0.45:
                by_cat[s.category.value] = by_cat.get(s.category.value, 0) + 1
        all_cats = {"structure", "aesthetic", "procedural", "doctrine", "anti"}
        return {
            "resolved": sorted(by_cat.keys()),
            "gaps": sorted(all_cats - set(by_cat)),
        }
