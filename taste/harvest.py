"""harvest — extracts provisional layers and conventions from task executions (P4/P5).

Learned patterns land strictly in provisional state in the inferred queue
awaiting owner confirmation (NEVER rule 2: agent amendment never activates itself).
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .core import StandardsEngine
from .models import Category, LayerState, Polarity, Provenance, Standard


class HarvestResult(BaseModel):
    harvested: int = 0
    queued: int = 0
    skipped: int = 0
    candidate_ids: list[str] = Field(default_factory=list)


def _extract_directives(text: str) -> list[tuple[Polarity, str]]:
    directives: list[tuple[Polarity, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line.startswith(("-", "*")):
            continue
        body = line.lstrip("-* ").strip()
        if not body:
            continue
        low = body.lower()
        if re.match(r"^(never|avoid|do not|don't)\b", low):
            directives.append((Polarity.NEVER, body))
        elif re.match(r"^(why|because)\b", low):
            directives.append((Polarity.WHY, body))
        elif re.match(r"^(always|prefer|lead|ensure|use)\b", low):
            directives.append((Polarity.DO, body))
    return directives


def harvest_task_run(
    engine: StandardsEngine,
    profile_id: str,
    task_ctx: dict[str, Any],
    output: str,
    rationale: str | None = None,
) -> HarvestResult:
    directives = _extract_directives(output)
    if not directives and rationale:
        directives = [(Polarity.WHY, f"In {task_ctx.get('type', 'task')}: {rationale}")]

    candidates: list[str] = []
    queued_count = 0
    for pol, text in directives:
        std = Standard(
            profile_id=profile_id,
            category=Category.ANTI if pol == Polarity.NEVER else Category.STRUCTURE,
            polarity=pol,
            directive=text,
            rationale=rationale or f"Harvested from task {task_ctx.get('type', 'run')}",
            state=LayerState.PROVISIONAL,
            provenance=Provenance.INFERENCE,
            variant_meta={"task_ctx": task_ctx},
        )
        engine.propose(std)
        candidates.append(std.id)
        queued_count += 1

    harvest_id = f"harv_{uuid4().hex[:8]}"
    engine.store.enqueue_inferred(
        harvest_id,
        profile_id,
        "harvest",
        {
            "task_ctx": task_ctx,
            "harvested_standards": candidates,
            "count": len(candidates),
        },
    )

    return HarvestResult(
        harvested=len(candidates),
        queued=queued_count,
        skipped=0,
        candidate_ids=candidates,
    )
