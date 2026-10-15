"""Tests for D1 rules scoring in taste-1."""

from __future__ import annotations

import pytest

from taste.core import StandardsEngine
from taste.models import Candidate, Category, LayerState, Polarity, Provenance, Standard
from taste.score import _forbidden_strings, score_candidates
from taste.store import Store


@pytest.fixture()
def engine(tmp_path):
    store = Store(tmp_path / "test.db")
    return StandardsEngine(store)


def _add_std(engine, directive, polarity=Polarity.NEVER, rationale="rule"):
    s = Standard(
        profile_id="owner", category=Category.ANTI if polarity == Polarity.NEVER else Category.STRUCTURE,
        polarity=polarity, directive=directive, rationale=rationale,
        state=LayerState.ACTIVE, provenance=Provenance.PERSONA,
    )
    from taste.embed import embed
    engine.store.put_standard(s, embed([directive + " " + rationale])[0])
    return s


def test_forbidden_strings_short_unquoted_rule():
    # 3 words after stripping "never": "use filler words"
    phrases = _forbidden_strings("Never use filler words")
    assert any("filler words" in p or "use filler words" in p for p in phrases)


def test_forbidden_strings_quoted_short_word():
    # Quoted word like "slop" or "buzz"
    phrases = _forbidden_strings('Never write "slop"')
    assert "slop" in phrases


def test_score_catches_short_unquoted_never_rule(engine):
    _add_std(engine, "Never use filler words")
    cands = [
        Candidate(label="A", content="This proposal uses filler words in each paragraph."),
        Candidate(label="B", content="Concise and clean engineering documentation."),
    ]
    scores = {s.ref: s for s in score_candidates(engine, "owner", {"type": "doc"}, cands)}
    assert len(scores["A"].violations) >= 1
    assert scores["A"].score < scores["B"].score
    assert len(scores["B"].violations) == 0


def test_score_breakdown_contains_required_metrics(engine):
    _add_std(engine, "Lead with architecture", polarity=Polarity.DO)
    _add_std(engine, "Never use generic openers", polarity=Polarity.NEVER)
    cands = [Candidate(label="A", content="In today's world, architecture matters.")]
    res = score_candidates(engine, "owner", {"type": "doc"}, cands)
    bd = res[0].breakdown
    assert "positive" in bd
    assert "negative" in bd
    assert "bloat_veto" in bd
    assert "leverage_fit" in bd
    assert "cost_tokens" in bd
    assert all(isinstance(v, float) for v in bd.values())
