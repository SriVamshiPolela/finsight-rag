from finsight import config
from finsight.llm.base import LLMProvider


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Returns the Anthropic or OpenAI provider based on config.LLM_PROVIDER
    (env var LLM_PROVIDER), demonstrating the vendor-swap pattern."""
    name = name or config.LLM_PROVIDER

    if name == "anthropic":
        from finsight.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=config.ANTHROPIC_API_KEY, model=config.ANTHROPIC_MODEL)

    if name == "openai":
        from finsight.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)

    raise ValueError(f"Unknown LLM_PROVIDER '{name}', expected 'anthropic' or 'openai'")
