"""driver — flash-tier cloud model dispatch for taste candidate generation (P4).

The model is just the driver — taste supplies the output. Config below is
the owner's spec from the TASTE heuristic research: high-conviction
inference params tuned against generic creative drift.

EXCLUDED BY OWNER DIRECTIVE: local generation models (Ollama, vLLM,
llama.cpp, any on-box LLM). Drivers are flash-tier API models only.
(Local embedding — MiniLM — is the C1 axis and unrelated to generation.)

API keys remain local in the DGF vault (~/Dev/.env, ~/.secrets/vault/) and
are read dynamically at runtime. Zero keys are ever written to git or logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("taste.driver")

# Inference parameters — owner's spec (Project Taste §4)
TEMPERATURE = 0.35        # against generic creative drift
TOP_P = 0.85              # high-conviction vocabulary only
PRESENCE_PENALTY = 0.10   # varied angles across candidates
REPETITION_PENALTY = 1.05 # no circular justification
MAX_TOKENS = 1024         # punchy outputs, no conversational bloat

# Flash-tier driver candidates — cloud APIs only, cheapest sufficient
DRIVER_CANDIDATES = [
    "qwen-flash",          # DashScope API — cloud, not local
    "deepseek-chat",
    "groq-4-fast",
    "gemini-2.5-flash",
]

_PROVIDER_URLS = {
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
}

_PROVIDER_MODELS = {
    "qwen": "qwen-flash",
    "dashscope": "qwen-flash",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "openrouter": "auto",
}

_PROVIDER_KEYS = {
    "qwen": ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
    "dashscope": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}

_HEDGE_WORDS = [
    "just", "really", "very", "basically", "actually",
    "in today's world", "it's important to note",
]


class DriverUnavailableError(RuntimeError):
    """No cloud provider API key found in env or DGF vault."""


def _candidate_vault_files() -> list[Path]:
    override = os.environ.get("TASTE_ENV_FILE")
    if override:
        return [Path(override)]
    return [
        Path.home() / "Dev" / ".env",
        Path.home() / "Dev" / ".secrets" / "vault" / "captain-snow.env",
        Path.home() / "Dev" / ".secrets" / "vault" / "root-master.env",
    ]


def resolve_api_key(provider: str) -> str | None:
    """Read provider key dynamically from process env or DGF vault files.

    Zero keys are stored in repo files or logged.
    """
    prov = provider.lower()
    names = _PROVIDER_KEYS.get(prov, [f"{prov.upper()}_API_KEY"])
    for name in names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()

    for p in _candidate_vault_files():
        try:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() in names:
                    clean = v.strip().strip("'\"")
                    if clean:
                        return clean
        except OSError:
            continue
    return None


def offline_driver(draft: str | None, rules: list[str] | None = None,
                   segment: str = "") -> list[str]:
    """Deterministic 3-candidate perturbation — zero spend, offline test safe."""
    base = (draft or "").strip()
    tight = base
    for w in _HEDGE_WORDS:
        tight = re.sub(rf"\b{re.escape(w)}\b[:,]?\s*", "", tight, flags=re.I)
    tight = re.sub(r"\s+", " ", tight).strip()

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", base) if s.strip()]
    reordered = " ".join(sorted(sents, key=len, reverse=True)) or base

    framed = base
    if rules:
        framed = f"{rules[0].rstrip('.')} — {base}"

    res = [tight or base or "Candidate A", reordered or base or "Candidate B", framed or base or "Candidate C"]
    return res


def parse_candidates_json(raw: str) -> list[str]:
    """Parse JSON array of candidate strings, tolerating markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: extract list from numbered or bulleted lines
    lines = [re.sub(r"^\s*(\d+[\.\)]|[-*])\s*", "", line).strip()
             for line in raw.splitlines() if line.strip()]
    return [line for line in lines if line]


def _pick_provider(requested: str | None) -> tuple[str, str] | None:
    if requested:
        key = resolve_api_key(requested)
        if key:
            return requested.lower(), key
        return None

    # Default cascade order across cloud providers in the vault
    order = ["qwen", "deepseek", "groq", "gemini", "openrouter"]
    for p in order:
        key = resolve_api_key(p)
        if key:
            return p, key
    return None


def call_driver(
    prompt: str,
    n: int = 3,
    provider: str | None = None,
    system: str | None = None,
    offline_fallback: bool = True,
    rules: list[str] | None = None,
    segment: str = "",
) -> list[str]:
    """Generate n candidates via the configured cloud flash driver.

    Falls back to deterministic offline generation if no key is present or on network error.
    """
    if os.environ.get("TASTE_OFFLINE") == "1":
        return offline_driver(prompt, rules, segment)[:n]

    pair = _pick_provider(provider)
    if not pair:
        if offline_fallback:
            return offline_driver(prompt, rules, segment)[:n]
        raise DriverUnavailableError("No cloud driver API key found in DGF vault or env")

    prov_name, api_key = pair
    base_url = _PROVIDER_URLS.get(prov_name, "https://api.openai.com/v1").rstrip("/")
    model = _PROVIDER_MODELS.get(prov_name, "qwen-flash")

    sys_content = system or (
        f"You are a concise, high-conviction writer following strict taste guardrails. "
        f"Generate exactly {n} distinct, high-quality candidates for the given segment. "
        f"Return ONLY a JSON array of {n} strings: [\"candidate 1\", ...]. No intro or commentary."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "presence_penalty": PRESENCE_PENALTY,
        "max_tokens": MAX_TOKENS,
    }

    try:
        import httpx

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            candidates = parse_candidates_json(raw_text)
            if candidates:
                return candidates[:n]
    except Exception as e:
        log.warning("Driver %s call failed: %s; falling back to offline generator", prov_name, e)
        if not offline_fallback:
            raise

    return offline_driver(prompt, rules, segment)[:n]
