"""triage — manages amendment proposals and review queue resolution (P4/P5).

All proposed amendments land in the review queue as provisional.
Only the owner can activate them by confirming with rationale.
"""

from __future__ import annotations

from typing import Any

from .core import StandardsEngine
from .models import Category, LayerState, Polarity, Provenance, Standard


def propose_amendment(
    engine: StandardsEngine,
    profile_id: str,
    directive: str,
    rationale: str,
    polarity: Polarity = Polarity.NEVER,
    category: Category = Category.STRUCTURE,
) -> Standard:
    """Propose an amendment. Strictly provisional; enqueued for owner review."""
    if not rationale or not rationale.strip():
        raise ValueError("rationale is required (NEVER rule 1)")

    std = Standard(
        profile_id=profile_id,
        category=category,
        polarity=polarity,
        directive=directive.strip(),
        rationale=rationale.strip(),
        state=LayerState.PROVISIONAL,
        provenance=Provenance.AMENDMENT,
    )
    engine.propose(std)

    queue_id = f"amend_{std.id}"
    engine.store.enqueue_inferred(
        queue_id,
        profile_id,
        "amendment",
        {
            "standard_id": std.id,
            "directive": std.directive,
            "rationale": std.rationale,
            "polarity": std.polarity.value,
        },
    )
    return std


def resolve_amendment(
    engine: StandardsEngine,
    iid: str,
    accept: bool,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Resolve an amendment in the queue. Accept activates; reject closes."""
    std_id = iid.removeprefix("amend_")
    if accept:
        engine.confirm(std_id, rationale=rationale)
        engine.store.resolve_inferred(iid)
        return {"status": "activated", "standard_id": std_id, "queue_id": iid}

    engine.store.resolve_inferred(iid)
    return {"status": "rejected", "standard_id": std_id, "queue_id": iid}


def list_triage_queue(engine: StandardsEngine, profile_id: str = "owner") -> list[dict[str, Any]]:
    """Return all pending inferred items awaiting review."""
    return engine.store.pending_inferred(profile_id)
