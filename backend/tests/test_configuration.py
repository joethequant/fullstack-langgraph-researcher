import os
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.graph import _init_model


def _mock_chat(name):
    class MockLLM:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    return MockLLM


@pytest.mark.parametrize(
    "model,env_key,expected_class",
    [
        ("gpt-4o", "OPENAI_API_KEY", "ChatOpenAI"),
        ("gemini-pro", "GEMINI_API_KEY", "ChatGoogleGenerativeAI"),
    ],
)
def test_from_runnable_config_selects_model(monkeypatch, model, env_key, expected_class):
    monkeypatch.setenv(env_key, "dummy")
    cfg = Configuration.from_runnable_config(RunnableConfig(configurable={"query_generator_model": model}))
    assert cfg.query_generator_model == model

    with patch(f"agent.graph.{expected_class}") as mocked_class:
        mocked_class.return_value = object()
        result = _init_model(cfg.query_generator_model, temperature=0.0)
        assert result is mocked_class.return_value
        mocked_class.assert_called_once()
