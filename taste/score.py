"""score — D1 permutation: rules scorer.

Two components, both free and deterministic:

  negative — lexical violation check: quoted strings inside `never`
             directives matched exactly; residual directive phrases checked
             by normalized containment. Each hit is a violation.
  positive — cosine of the candidate against the centroid of relevant
             `do`/`why` layers + exemplars: does it sit inside the
             taste-shaped region?

score = positive − 0.5 × violations, clamped to [−1, 1].
"""

from __future__ import annotations

import re

from .core import StandardsEngine
from .embed import cosine, embed
from .models import Candidate, CandidateScore, Polarity

QUOTED = re.compile(
    r'"([^"]{2,})"|“([^”]{2,})”'      # double / smart quotes
    r"|(?<![\w'])'([^']{2,})'(?!\w)"  # single quotes, apostrophe-safe
)


def _quoted_strings(text: str) -> list[str]:
    return [next(g for g in m if g) for m in QUOTED.findall(text)]


LEADIN = re.compile(r"^(never|avoid|do not|don't|no|stop|refrain from)\s+", re.I)
SUB_ACTION = re.compile(r"^(use|write|include|say|post|generate|employ)\s+", re.I)


def _forbidden_strings(directive: str) -> list[str]:
    """Extract the concrete forbidden material from a never-directive."""
    quoted = _quoted_strings(directive)
    stripped = LEADIN.sub("", directive).strip().rstrip(".")
    phrases = list(quoted)
    if stripped:
        phrases.append(stripped)
        sub = SUB_ACTION.sub("", stripped).strip()
        if sub and sub != stripped and len(sub.split()) >= 1:
            phrases.append(sub)
    # Deduplicate while preserving order
    seen: set[str] = set()
    res: list[str] = []
    for p in phrases:
        low = p.lower()
        if low and low not in seen:
            seen.add(low)
            res.append(low)
    return res


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def score_candidates(engine: StandardsEngine, profile_id: str,
                     task_ctx: dict, candidates: list[Candidate]
                     ) -> list[CandidateScore]:
    query = " ".join(str(v) for v in task_ctx.values() if isinstance(v, str))
    hits = engine.retrieve(query or "general", profile_id)

    # guardrails apply unconditionally: every active never-rule is checked
    nevers = [s for s in engine.active(profile_id)
              if s.polarity == Polarity.NEVER]
    positives = [s for s, sim in hits
                 if s.polarity in (Polarity.DO, Polarity.WHY) and sim > 0.3]

    centroid = None
    if positives:
        vecs = embed([s.directive + " " + s.rationale for s in positives])
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    out: list[CandidateScore] = []
    for cand in candidates:
        text = cand.content or cand.content_ref
        norm = _norm(text)

        violations: list[str] = []
        for s in nevers:
            for phrase in _forbidden_strings(s.directive):
                if phrase and phrase in norm:
                    violations.append(f"violates {s.id}: '{phrase}'")

        positive = 0.0
        if centroid is not None and text.strip():
            positive = max(0.0, cosine(embed([text])[0], centroid))

        score = max(-1.0, min(1.0, positive - 0.5 * len(violations)))
        out.append(CandidateScore(
            ref=cand.label, score=round(score, 4),
            # owner's audit matrix as named dims (contract keeps
            # positive/negative as the mechanical minimum):
            #   bloat_veto   — violation pressure (anti-pattern hits)
            #   leverage_fit — sits inside the taste-shaped region
            #   cost_tokens  — output size ≈ marginal cost proxy
            #   punch        — owner-judged until the learned scorer (D3)
            #                  absorbs it from decisions; not faked here
            breakdown={
                "positive": round(positive, 4),
                "negative": float(len(violations)),
                "bloat_veto": float(len(violations)),
                "leverage_fit": round(positive, 4),
                "cost_tokens": float(len(text) // 4),
            },
            violations=violations))
    return out
