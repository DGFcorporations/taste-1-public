"""inject — I1 permutation: packet compiler → directive blocks.

The packet IS the product: predetermined structure, directives, exemplars —
the model only fills connective tissue. Deterministic: same corpus + same
input → same packet (contract §4).
"""

from __future__ import annotations

from .core import StandardsEngine
from .models import (
    Constraint,
    Directive,
    Exemplar,
    Packet,
    PacketRequest,
    Polarity,
)

import re

SIM_THRESHOLD = 0.40          # a do/why layer must be relevant to apply
MAX_DIRECTIVES = 8            # packet size is bounded — every token is cost
MAX_NEVERS = 4                # guardrails always apply, bounded
MAX_EXEMPLARS = 2

# Domain synonym keywords for boosting relevance
DOMAIN_KEYWORDS = {
    "media": ["video", "audio", "mp4", "render", "youtube", "tiktok", "reels", "voice", "sync", "lip", "thumbnail", "kallaway", "hormozi", "elevenlabs", "qwen", "deepgram", "camera", "frame", "cut"],
    "content": ["copy", "text", "headline", "hook", "story", "framing", "script", "editorial", "email", "angle", "jargon", "fluff"],
    "engineering": ["code", "bug", "fix", "feature", "git", "commit", "test", "tdd", "branch", "refactor", "security", "fail-closed", "clean", "architecture"],
    "systems": ["container", "docker", "service", "systemd", "server", "port", "daemon", "proxy", "cloudflare", "postgres", "redis", "bridge", "rclone"],
    "governance": ["gate", "audit", "compliance", "phi", "legal", "secret", "env", "credential", "permission", "sterile-field"],
}


def score_standard_relevance(s, query: str, tokens: set[str], hit_score: float = 0.0) -> float:
    text = (s.directive + " " + (s.rationale or "")).lower()
    score = hit_score * 10.0
    for t in tokens:
        if len(t) > 3 and t in text:
            score += 3.0
        elif t in text:
            score += 1.0
    q_low = query.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in q_low for k in kws):
            if any(k in text for k in kws):
                score += 5.0
    conf = float(s.confidence if s.confidence is not None else 0.5)
    return score * (0.5 + conf)


def compile_packet(engine: StandardsEngine,
                   req: PacketRequest) -> Packet:
    query = " ".join(
        str(v) for v in req.task_ctx.values() if isinstance(v, str))
    for ans in req.answers.values():
        query += " " + ans
    query = query.strip() or "general task"

    hits = engine.retrieve(query, req.profile_id)
    hit_dict = {s.id: sim for s, sim in hits}
    tokens = set(re.findall(r"\w+", query.lower()))

    # Rank NEVER rules by task relevance + confidence, break ties by s.id for determinism
    active_nevers = [
        s for s in engine.active(req.profile_id)
        if s.polarity == Polarity.NEVER
    ]
    ranked_nevers = sorted(
        active_nevers,
        key=lambda s: (score_standard_relevance(s, query, tokens, hit_dict.get(s.id, 0.0)), s.id),
        reverse=True,
    )
    selected_nevers = sorted(ranked_nevers[:MAX_NEVERS], key=lambda s: s.id) if ranked_nevers else []
    directives: list[Directive] = [
        Directive(polarity=s.polarity, text=s.directive, standard_id=s.id)
        for s in selected_nevers]

    exemplars: list[Exemplar] = []
    # Rank hits by score and relevance
    ranked_hits = sorted(
        hits,
        key=lambda item: (score_standard_relevance(item[0], query, tokens, item[1]), item[1]),
        reverse=True,
    )
    for s, sim in ranked_hits:
        if s.polarity == Polarity.NEVER:
            continue  # already sourced above — guardrails are not hits
        if sim < SIM_THRESHOLD:
            continue
        if s.polarity == Polarity.WHY:
            if len(exemplars) < MAX_EXEMPLARS:
                exemplars.append(Exemplar(text=s.directive, why=s.rationale))
            continue
        if len(directives) >= MAX_DIRECTIVES:
            break
        directives.append(Directive(
            polarity=s.polarity, text=s.directive, standard_id=s.id))

    # owner-supplied task guidelines land as hard constraints — they outrank
    # accumulated taste for THIS task (fresh signal law)
    constraints = [Constraint(hard=True, text=g) for g in req.guidelines]

    # NEVER rule 6 — one bounded free degree minimum, always
    free_degrees = [
        "voice and rhythm within the directives — vary phrasing, "
        "keep the pattern alive"]

    return Packet(
        profile_id=req.profile_id, task_ctx=req.task_ctx,
        directives=directives, exemplars=exemplars,
        constraints=constraints, free_degrees=free_degrees,
        variant_meta={"engine": "taste-1", "hits": len(hits)})
