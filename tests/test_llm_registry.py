from unittest.mock import patch

import pytest

from finsight import config
from finsight.llm.registry import get_llm_provider


def test_defaults_to_anthropic(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "fake-key")
    with patch("finsight.llm.anthropic_provider.anthropic.Anthropic"):
        provider = get_llm_provider()
    assert provider.name == "anthropic"


def test_dispatches_to_openai_when_configured(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-key")
    with patch("finsight.llm.openai_provider.openai.OpenAI"):
        provider = get_llm_provider("openai")
    assert provider.name == "openai"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_llm_provider("not-a-real-provider")
