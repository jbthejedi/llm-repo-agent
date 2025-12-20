import pytest

from llm_repo_agent.actions import parse_action, ToolCallAction, FinalAction, ActionParseError


def test_parse_tool_call():
    obj = {"type": "tool_call", "name": "list_files", "args": {"rel_dir": ".", "max_files": 10}, "thought": "scan root"}
    a = parse_action(obj)
    assert isinstance(a, ToolCallAction)
    assert a.name == "list_files"
    assert a.thought == "scan root"
    assert "thought" not in a.args


def test_coerce_type_to_tool_call():
    obj = {"type": "list_files", "args": {"rel_dir": ".", "max_files": 10}}
    a = parse_action(obj)
    assert isinstance(a, ToolCallAction)
    assert a.name == "list_files"
    assert a.thought is None


def test_parse_final():
    obj = {"type": "final", "summary": "Done", "changes": [], "thought": "nothing left"}
    a = parse_action(obj)
    assert isinstance(a, FinalAction)
    assert a.summary == "Done"
    assert a.thought == "nothing left"


def test_invalid_tool_call_raises():
    obj = {"type": "tool_call", "name": "", "args": {}}
    with pytest.raises(ActionParseError):
        parse_action(obj)


def test_invalid_type_raises():
    with pytest.raises(ActionParseError):
        parse_action("not a dict")


def test_thought_must_be_string_if_present():
    obj = {"type": "tool_call", "name": "read_file", "args": {"rel_path": "x", "max_chars": 10}, "thought": 123}
    with pytest.raises(ActionParseError):
        parse_action(obj)


def test_thought_can_ride_in_args_and_is_stripped():
    obj = {"type": "tool_call", "name": "read_file", "args": {"rel_path": "x", "max_chars": 10, "thought": "peek"}}
    a = parse_action(obj)
    assert isinstance(a, ToolCallAction)
    assert a.thought == "peek"
    assert "thought" not in a.args
