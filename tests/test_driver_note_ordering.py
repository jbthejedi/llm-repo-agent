import copy
from types import SimpleNamespace

from llm_repo_agent.llm import ChatCompletionsLLM
from llm_repo_agent.actions import ToolCallAction, FinalAction


class DummyClient:
    def __init__(self, responses, recorded_messages):
        self._responses = list(responses)
        self._recorded = recorded_messages
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self._recorded.append(copy.deepcopy(kwargs.get("messages", [])))
        return self._responses.pop(0)


def _tool_call_message(name, arguments, call_id="call_1"):
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(tool_calls=[tool_call], content="")


def _final_message(content):
    return SimpleNamespace(tool_calls=[], content=content)


def test_driver_note_appended_after_tool_result():
    recorded = []
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=_tool_call_message(
            "read_file", "{\"rel_path\":\"python_programs/gcd.py\",\"max_chars\":1000}"
        ))]),
        SimpleNamespace(choices=[SimpleNamespace(message=_final_message(
            "{\"type\":\"final\",\"summary\":\"ok\",\"changes\":[],\"thought\":\"\"}"
        ))]),
    ]

    llm = ChatCompletionsLLM(model="gpt-4.1-mini")
    llm.client = DummyClient(responses, recorded)
    llm.start_conversation("sys", "goal")

    first = llm.next_action()
    assert isinstance(first, ToolCallAction)

    llm.add_driver_note("check test path")
    second = llm.next_action("file contents")
    assert isinstance(second, FinalAction)

    assert len(recorded) == 2
    messages = recorded[1]
    roles = [m.get("role") for m in messages]
    assert roles[-3:] == ["assistant", "tool", "system"]
    assert "tool_calls" in messages[-3]
    assert messages[-1]["content"].startswith("DRIVER NOTE:\ncheck test path")
