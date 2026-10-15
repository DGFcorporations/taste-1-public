"""Tests for taste/harvest.py and taste/triage.py (P4/P5 harvest & amendment pipelines)."""

from __future__ import annotations

import pytest

from taste.core import StandardsEngine
from taste.harvest import HarvestResult, harvest_task_run
from taste.models import LayerState, Polarity, Provenance
from taste.store import Store
from taste.triage import list_triage_queue, propose_amendment, resolve_amendment


@pytest.fixture()
def engine(tmp_path):
    store = Store(tmp_path / "test.db")
    return StandardsEngine(store)


def test_propose_amendment_lands_provisional_and_queued(engine):
    std = propose_amendment(
        engine,
        profile_id="owner",
        directive="Never emit conversational pleasantries like 'Sure thing!'",
        rationale="Punchy technical communications only",
        polarity=Polarity.NEVER,
    )
    # 1. Must be provisional
    assert std.state == LayerState.PROVISIONAL
    assert std.provenance == Provenance.AMENDMENT

    # 2. Must be stored in standards
    stored = engine.store.get_standard(std.id)
    assert stored is not None
    assert stored.state == LayerState.PROVISIONAL

    # 3. Must be queued in inferred_queue
    queue = list_triage_queue(engine, "owner")
    assert any(q["id"] == f"amend_{std.id}" or q["payload"].get("standard_id") == std.id for q in queue)


def test_propose_amendment_rejects_blank_rationale(engine):
    with pytest.raises(ValueError, match="rationale"):
        propose_amendment(
            engine,
            profile_id="owner",
            directive="Never say oops",
            rationale="   ",
        )


def test_resolve_amendment_activation_requires_owner_confirmation(engine):
    std = propose_amendment(
        engine,
        profile_id="owner",
        directive="Never hardcode endpoints",
        rationale="All endpoints must resolve via service discovery",
    )
    assert engine.store.get_standard(std.id).state == LayerState.PROVISIONAL

    # Resolving with accept=True confirms and activates
    res = resolve_amendment(engine, f"amend_{std.id}", accept=True)
    assert res["status"] == "activated"
    assert engine.store.get_standard(std.id).state == LayerState.ACTIVE


def test_resolve_amendment_rejection_leaves_it_unactivated(engine):
    std = propose_amendment(
        engine,
        profile_id="owner",
        directive="Never use yaml",
        rationale="JSON only",
    )
    res = resolve_amendment(engine, f"amend_{std.id}", accept=False)
    assert res["status"] == "rejected"
    assert engine.store.get_standard(std.id).state == LayerState.PROVISIONAL


def test_harvest_task_run_enqueues_provisional_findings(engine):
    task_ctx = {"type": "deploy_service", "service": "indexer"}
    output = """
    # Run Log
    - Never deploy without running database migration pre-check.
    - Always verify health endpoint responds with 200 within 5 seconds.
    """
    res = harvest_task_run(engine, "owner", task_ctx, output, rationale="Automated deployment pattern")
    assert isinstance(res, HarvestResult)
    assert res.harvested >= 1
    assert res.queued >= 1

    # Inferred queue has the harvest event
    queue = list_triage_queue(engine, "owner")
    assert any(q["kind"] == "harvest" for q in queue)

    # Any created standard is provisional
    stds = engine.store.list_standards("owner", state=LayerState.ACTIVE)
    assert len(stds) == 0, "Harvested standards must NEVER auto-activate!"
