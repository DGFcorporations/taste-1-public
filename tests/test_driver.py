"""Tests for taste/driver.py: local DGF vault key resolution, dispatch, and offline fallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from taste.driver import (
    DriverUnavailableError,
    call_driver,
    offline_driver,
    parse_candidates_json,
    resolve_api_key,
)


def test_offline_driver_returns_deterministic_candidates():
    draft = "In today's world, we really just need fast reliable systems."
    rules = ["Lead with the architecture"]
    res = offline_driver(draft, rules, "body")
    assert len(res) == 3
    assert all(isinstance(c, str) and c.strip() for c in res)
    # Checks hedges stripped in first candidate
    assert "in today's world" not in res[0].lower()
    assert "really" not in res[0].lower()
    # Checks framed directive in candidate 3
    assert "Lead with the architecture" in res[2]


def test_parse_candidates_json_handles_fenced_and_raw():
    raw_fenced = '```json\n["Option A", "Option B", "Option C"]\n```'
    assert parse_candidates_json(raw_fenced) == ["Option A", "Option B", "Option C"]

    raw_dict = '{"candidates": ["Cand 1", "Cand 2"]}'
    assert parse_candidates_json(raw_dict) == ["Cand 1", "Cand 2"]

    raw_lines = "1. First candidate\n2. Second candidate\n3. Third candidate"
    assert parse_candidates_json(raw_lines) == ["First candidate", "Second candidate", "Third candidate"]


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    key = resolve_api_key("groq")
    assert key == "test-groq-key"


def test_resolve_api_key_from_vault_file(tmp_path, monkeypatch):
    vault = tmp_path / "test.env"
    vault.write_text("QWEN_API_KEY=test-qwen-vault-key\n")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("TASTE_ENV_FILE", str(vault))

    key = resolve_api_key("qwen")
    assert key == "test-qwen-vault-key"


def test_call_driver_offline_fallback_when_no_key(monkeypatch):
    monkeypatch.setenv("TASTE_OFFLINE", "1")
    res = call_driver("Draft text", n=3)
    assert len(res) == 3
    assert all(isinstance(c, str) for c in res)


def test_call_driver_raises_when_no_key_and_no_fallback(monkeypatch, tmp_path):
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    monkeypatch.setenv("TASTE_ENV_FILE", str(empty_env))
    for k in ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)

    with pytest.raises(DriverUnavailableError):
        call_driver("Draft text", n=3, provider="deepseek", offline_fallback=False)


def test_call_driver_mock_live_dispatch(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": '["Mock candidate 1", "Mock candidate 2", "Mock candidate 3"]'
            }
        }]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        cands = call_driver("Test prompt", n=3, provider="groq")
        assert cands == ["Mock candidate 1", "Mock candidate 2", "Mock candidate 3"]
