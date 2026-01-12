from types import SimpleNamespace

import pytest

from llm_repo_agent.actions import ToolCallAction, FinalAction, ActionParseError
from llm_repo_agent.llm import JsonToolLLM


class DummyClient:
    def __init__(self, responses, recorded):
        self._responses = list(responses)
        self._recorded = recorded
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self._recorded.append(kwargs)
        return self._responses.pop(0)


def _message(content):
    return SimpleNamespace(content=content, tool_calls=[])


def test_json_tool_llm_parses_tool_calls_and_appends_tool_result():
    recorded = []
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=_message(
            "{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{\"rel_path\":\"a.py\",\"max_chars\":10}}"
        ))]),
        SimpleNamespace(choices=[SimpleNamespace(message=_message(
            "{\"type\":\"final\",\"summary\":\"ok\",\"changes\":[],\"thought\":\"\"}"
        ))]),
    ]

    llm = JsonToolLLM(model="gpt-4.1-mini")
    llm.client = DummyClient(responses, recorded)
    llm.start_conversation("sys", "GOAL:\nFix it")

    first = llm.next_action()
    assert isinstance(first, ToolCallAction)
    assert "tools" not in recorded[0]

    second = llm.next_action("file contents")
    assert isinstance(second, FinalAction)
    assert llm._messages[-2]["role"] == "user"
    assert llm._messages[-2]["content"].startswith("[tool_result]\nfile contents")


def test_json_tool_llm_logs_invalid_json():
    recorded = []
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=_message("not-json"))]),
    ]

    llm = JsonToolLLM(model="gpt-4.1-mini")
    llm.client = DummyClient(responses, recorded)
    llm.start_conversation("sys", "GOAL:\nFix it")

    with pytest.raises(ActionParseError):
        llm.next_action()

    assert llm._last_raw["type"] == "invalid_json"
    assert "raw_text" in llm._last_raw
