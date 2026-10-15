"""hone — F1 permutation: best-of-3, taste-scored.

Generate 3 candidates → score → pick → iterate flag if under threshold.
The driver (candidate generator) is pluggable: real wiring calls the flash
model per segment; the offline default perturbs the draft deterministically
so the loop is fully testable without model spend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from .core import StandardsEngine
from .driver import call_driver
from .inject import compile_packet
from .models import (
    Candidate,
    HoneRequest,
    HoneResponse,
    PacketRequest,
)
from .score import score_candidates

SCORE_THRESHOLD = 0.55
N_CANDIDATES = 3
HONE_LIMIT = 15


def task_key(ctx: dict) -> str:
    """Segments of one task share a hone budget. Attendance does not reset it."""
    for field in ("task_ref", "ref"):
        value = ctx.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    relevant = {key: value for key, value in ctx.items() if key not in {"segment", "attended"}}
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return "task_" + hashlib.sha256(canonical.encode()).hexdigest()

Driver = Callable[[str | None, dict, list[str]], list[str]]


def hone_segment(engine: StandardsEngine, req: HoneRequest,
                 driver: Driver | None = None) -> HoneResponse:
    reservation = engine.store.reserve_hone(
        req.profile_id, task_key(req.segment_ctx), HONE_LIMIT)
    ctx = {**req.segment_ctx, "attended": False}
    packet = compile_packet(engine, PacketRequest(
        profile_id=req.profile_id, task_ctx=ctx))
    do_rules = [d.text for d in packet.directives
                if d.polarity.value == "do"]

    if driver is not None:
        raw = driver(req.draft, req.segment_ctx, do_rules)
    else:
        raw = call_driver(
            req.draft or "",
            n=N_CANDIDATES,
            rules=do_rules,
            segment=str(req.segment_ctx.get("segment", "")),
        )

    candidates = [Candidate(label=chr(65 + i), content=t)
                  for i, t in enumerate(raw[:N_CANDIDATES])]
    if not candidates:
        # driver returned nothing — an empty result is not an error
        return HoneResponse(candidates=[], pick="", score=0.0,
                            iterate=False)
    scored = score_candidates(engine, req.profile_id, req.segment_ctx,
                              candidates)
    if not scored:
        return HoneResponse(candidates=candidates, pick="", score=0.0,
                            iterate=False)
    best = max(scored, key=lambda s: s.score)

    return HoneResponse(
        candidates=candidates, pick=best.ref, score=best.score,
        iterate=best.score < SCORE_THRESHOLD and reservation < HONE_LIMIT,
    )
