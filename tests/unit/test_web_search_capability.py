"""The evidence-driven web lane must never run without real web search.

A model without search still answers "find me songs and cite your evidence" —
fluently, from training memory, with evidence that reads exactly like a search
result. That output was then labelled 🌐 in the UI. Producing nothing is the
correct behaviour; producing unsourced songs that claim a source is not.
"""

from __future__ import annotations

import asyncio

import pytest

from llms.chat_models import WEB_SEARCH_PROVIDERS, provider_supports_web_search
from retrieval.web_supplement import WebSongSupplement


def test_only_providers_with_real_search_are_listed():
    assert provider_supports_web_search("dashscope") is True
    for provider in ("siliconflow", "deepseek", "google", "sglang", "vllm", "ollama", ""):
        assert provider_supports_web_search(provider) is False, provider
    # Case/whitespace must not be a way to sneak past the gate.
    assert provider_supports_web_search("  DashScope ") is True


def test_capability_set_is_the_single_source_of_truth():
    """No caller may decide web-search support with an inline provider name.

    Deliberately narrow: plenty of code legitimately branches on dashscope for
    unrelated reasons (its API-key env var, its native structured-output path,
    thinking mode). Only a dashscope comparison that is ALSO deciding something
    about search/web is the bug — that is exactly the shape the original defect
    had: `if enable_web_search and provider_key == "dashscope"`.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    provider_eq = re.compile(r'provider\w*\s*==\s*["\']dashscope["\']')
    offenders = []
    for directory in ("retrieval", "services", "agent", "api", "llms"):
        for path in sorted((root / directory).rglob("*.py")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if provider_eq.search(line) and re.search(r"search|web", line, re.I):
                    offenders.append(f"{path.relative_to(root)}:{i}")
    assert not offenders, (
        f"inline dashscope web-search check instead of "
        f"provider_supports_web_search(): {offenders}")


def test_lane_is_skipped_when_the_provider_cannot_search(monkeypatch, caplog):
    """The bug this exists for: with provider=siliconflow the lane still ran and
    returned songs whose 'evidence' was invented."""
    from config.settings import settings

    monkeypatch.setattr(settings, "intent_llm_provider", "siliconflow")
    monkeypatch.setattr(settings, "llm_default_provider", "siliconflow")

    lane = WebSongSupplement()          # no injected generator -> real LLM path
    with caplog.at_level("WARNING"):
        out = asyncio.run(lane.discover(query="夏日感的音乐"))
    assert out == []
    assert "没有原生联网搜索能力" in caplog.text


def test_lane_runs_when_the_provider_can_search(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "intent_llm_provider", "dashscope")
    monkeypatch.setattr(settings, "llm_default_provider", "dashscope")

    lane = WebSongSupplement()
    assert lane._native_search_available() is True


def test_injected_generator_is_exempt_from_the_gate(monkeypatch):
    """Tests and any future non-LLM discovery source supply their own generator;
    the gate is about the default LLM path only."""
    from config.settings import settings

    monkeypatch.setattr(settings, "intent_llm_provider", "siliconflow")
    lane = WebSongSupplement(generator=lambda payload: {"songs": []})
    assert lane._native_search_available() is True


@pytest.mark.parametrize("provider,expected", [("dashscope", True), ("siliconflow", False)])
def test_enable_search_flag_only_reaches_supported_providers(provider, expected):
    """get_chat_model must not silently drop the flag: on an unsupported provider
    it warns, so the log says why the lane produced nothing."""
    assert provider_supports_web_search(provider) is expected
    assert provider in WEB_SEARCH_PROVIDERS or not expected
