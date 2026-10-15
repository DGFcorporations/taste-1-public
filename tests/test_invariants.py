"""Invariant and regression tests for the control variant.

Each test pins one of the seven load-bearing design rules (docs/BLUEPRINT.md)
or a defect found in review. Offline: hashing embedder via conftest.
"""

import pytest
from pydantic import ValidationError

from taste.models import (
    Category,
    DecisionRecord,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)


def _std(directive, polarity=Polarity.NEVER, rationale="because taste",
         state=LayerState.ACTIVE, provenance=Provenance.PERSONA):
    return Standard(profile_id="owner", category=Category.AESTHETIC,
                    polarity=polarity, directive=directive,
                    rationale=rationale, state=state, provenance=provenance)


def _import(client, standards):
    """Import, then locally confirm rows the test submitted as active.

    A declared active state is not approval. Tests of veto and packet
    behavior confirm the exact row first. Quarantine itself is pinned
    in test_import_approval.py.
    """
    resp = client.post("/standards/import", json={
        "profile_id": "owner",
        "standards": [s.model_dump(mode="json") for s in standards]})
    rows = {r["directive"]: r for r in
            client.get("/standards").json()["standards"]}
    for s in standards:
        if (s.state == LayerState.ACTIVE and s.rationale.strip()
                and s.provenance not in (Provenance.AMENDMENT, Provenance.INFERENCE)):
            row = rows[s.directive]
            client.post(f"/standards/{row['id']}/confirm",
                        json={"rationale": s.rationale})
    return resp


# Rule 3 — rationale required for decisions -------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_decision_record_rejects_blank_rationale(bad):
    with pytest.raises(ValidationError):
        DecisionRecord(pick="A", rationale=bad)


def test_decision_record_accepts_real_rationale():
    assert DecisionRecord(pick="A", rationale="verdict-first won").pick == "A"


# Import dedup (contract §3) ----------------------------------------------
def test_import_dedups_within_a_batch(client):
    s = _std("Never open with a generic hook")
    body = _import(client, [s, s]).json()
    assert body == {"imported": 1, "skipped": 1}
    assert client.get("/standards").json()["count"] == 1


def test_import_dedups_against_existing_corpus(client):
    s = _std("Never open with a generic hook")
    _import(client, [s])
    assert _import(client, [s]).json() == {"imported": 0, "skipped": 1}


# Rule 2/3 — import never activates a layer lacking rationale -------------
def test_import_without_rationale_stays_provisional(client):
    s = _std("Never pad with filler", rationale="")
    _import(client, [s])
    row = client.get("/standards").json()["standards"][0]
    assert row["state"] == "provisional"


# Elimination filter: coverage is measured against the RULE's terms --------
def test_elimination_filter_flags_veto_amid_extra_task_words(client):
    _import(client, [_std("Never use generic openers in newsletters")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief":
                     "Write an article using generic openers for our "
                     "newsletter audience about quarterly shipping notes"}})
    assert r.json()["vetoes_hit"], "never-rule hit must not be diluted"


def test_elimination_filter_quiet_on_unrelated_task(client):
    _import(client, [_std("Never use generic openers in newsletters")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "quarterly invoice summary"}})
    assert r.json()["vetoes_hit"] == []


# Rule 4 — packet determinism ---------------------------------------------
def test_packet_is_deterministic(client):
    _import(client, [_std("Never use generic openers"),
                     _std("Lead with the angle", Polarity.DO)])
    req = {"profile_id": "owner",
           "task_ctx": {"type": "write_doc", "brief": "ship notes"}}
    a = client.post("/packet", json=req).json()
    b = client.post("/packet", json=req).json()
    strip = lambda p: {k: v for k, v in p.items()  # noqa: E731
                       if k not in ("id", "ts", "packet_id", "compiled_at")}
    assert strip(a) == strip(b)


# Rule 1 — never-rules always apply ---------------------------------------
def test_never_rules_apply_even_when_irrelevant_to_task(client):
    _import(client, [_std("Never mention competitors by name")])
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "zzz unrelated qqq"}}).json()
    assert any(d["polarity"] == "never" for d in pkt["directives"])


# Rule 2 — propose never applies ------------------------------------------
def test_ingested_layers_arrive_provisional(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "- Never use filler words.\n", "format": "md"}]})
    states = {s["state"] for s in client.get("/standards").json()["standards"]}
    assert states <= {"provisional"}


def test_confirm_without_rationale_is_refused(app):
    from taste.core import StandardsEngine  # noqa: F401
    engine = app.state.engine
    s = engine.propose(_std("Never pad", rationale="", state=LayerState.PROVISIONAL))
    with pytest.raises(ValueError):
        engine.confirm(s.id)
    assert app.state.store.get_standard(s.id).state == LayerState.PROVISIONAL


# CLI: --state filter is honoured -----------------------------------------
def test_cli_standards_state_filter(app, capsys, monkeypatch):
    from taste import cli
    engine = app.state.engine
    engine.propose(_std("Never pad prose"))
    monkeypatch.setattr(cli, "create_app", lambda: app)
    cli.main(["standards", "--state=active"])
    assert "Never pad prose" not in capsys.readouterr().out
    cli.main(["standards", "--state=provisional"])
    assert "Never pad prose" in capsys.readouterr().out


# Rule 1 — rationale must carry content over the wire --------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_feedback_blank_rationale_rejected_422(client, bad):
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "x"},
        "pick": "A", "rationale": bad})
    assert r.status_code == 422, f"rationale={bad!r} must 422"


# CODE-002 — never-rules come from active corpus, not retrieval hits -----
def test_never_rule_absent_from_retrieval_still_in_packet():
    """A never-rule that retrieval never returns must still guard the packet."""
    from taste.inject import compile_packet
    from taste.models import PacketRequest

    never = _std("Never mention competitors by name")

    class _StubEngine:
        def retrieve(self, query, profile_id="owner", **kw):
            return []  # retrieval returns nothing at all
        def active(self, profile_id="owner"):
            return [never]

    pkt = compile_packet(_StubEngine(), PacketRequest(
        profile_id="owner", task_ctx={"type": "write_doc",
                                      "brief": "unrelated task"}))
    assert [d.text for d in pkt.directives] == [never.directive]
    assert pkt.directives[0].polarity == Polarity.NEVER


def test_packet_never_rules_deterministic_order(client):
    _import(client, [_std("Never mention competitors by name"),
                     _std("Never use generic openers"),
                     _std("Never pad with filler")])
    req = {"profile_id": "owner",
           "task_ctx": {"type": "write_doc", "brief": "ship notes"}}
    a = client.post("/packet", json=req).json()
    b = client.post("/packet", json=req).json()
    nevers_a = [d["text"] for d in a["directives"] if d["polarity"] == "never"]
    nevers_b = [d["text"] for d in b["directives"] if d["polarity"] == "never"]
    assert nevers_a == nevers_b
    assert len(nevers_a) == 3


# Veto formula — long rule diluted by short task; short rule --------------
def test_veto_fires_when_rule_terms_dominate_short_task(client):
    """Long never-rule, short brief mostly made of its terms -> veto."""
    _import(client, [_std(
        "Never use generic openers like in today's world as widely "
        "known phrases in copywriting deliverables")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "generic openers today"}})
    assert r.json()["vetoes_hit"], "task_cov clause must fire"


def test_veto_fires_on_short_single_term_rule(client):
    _import(client, [_std("No emoji")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "post", "brief": "add emoji everywhere"}})
    assert r.json()["vetoes_hit"], "single-term rule must fire"


def test_veto_quiet_when_coverage_below_threshold(client):
    _import(client, [_std(
        "Never use generic openers like in today's world as widely "
        "known phrases in copywriting deliverables")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc",
                     "brief": "quarterly invoice summary report"}})
    assert r.json()["vetoes_hit"] == []


# API minor fixes ---------------------------------------------------------
def test_standards_invalid_state_returns_422_error_body(client):
    r = client.get("/standards", params={"state": "bad"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "invalid_state"


def test_supersede_nonexistent_returns_404(client):
    r = client.post("/standards/std_missing/supersede", json={
        "profile_id": "owner", "category": "doctrine", "polarity": "do",
        "directive": "Fresh doctrine", "rationale": "better"})
    assert r.status_code == 404


def test_hone_empty_driver_returns_empty_response(tmp_path):
    from taste.core import StandardsEngine
    from taste.hone import hone_segment
    from taste.models import HoneRequest
    from taste.store import Store

    engine = StandardsEngine(Store(tmp_path / "h.db"))
    resp = hone_segment(engine, HoneRequest(
        profile_id="owner", segment_ctx={"segment": "intro"}),
        driver=lambda draft, ctx, rules: [])
    assert resp.candidates == []
    assert resp.iterate is False
