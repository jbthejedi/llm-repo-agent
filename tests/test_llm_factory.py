import pytest

from llm_repo_agent.llm import LLMFactory, LLMConfig, ChatCompletionsLLM


def test_factory_builds_openai():
    cfg = LLMConfig(provider="openai", model="gpt-4.1-mini")
    llm = LLMFactory.build(cfg)
    assert isinstance(llm, ChatCompletionsLLM)
    assert llm.model == "gpt-4.1-mini"
    assert llm.base_url is None


def test_factory_builds_together():
    cfg = LLMConfig(provider="together", model="mistralai/Mistral-7B-Instruct-v0.3")
    llm = LLMFactory.build(cfg)
    assert isinstance(llm, ChatCompletionsLLM)
    assert llm.model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert llm.base_url == "https://api.together.xyz/v1"


def test_factory_rejects_unknown_provider():
    cfg = LLMConfig(provider="nope")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.build(cfg)
