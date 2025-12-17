import pytest

from llm_repo_agent.actions import parse_action, ToolCallAction, FinalAction, ActionParseError


def test_parse_tool_call():
    obj = {"type": "tool_call", "name": "list_files", "args": {"rel_dir": ".", "max_files": 10}}
    a = parse_action(obj)
    assert isinstance(a, ToolCallAction)
    assert a.name == "list_files"


def test_coerce_type_to_tool_call():
    obj = {"type": "list_files", "args": {"rel_dir": ".", "max_files": 10}}
    a = parse_action(obj)
    assert isinstance(a, ToolCallAction)
    assert a.name == "list_files"


def test_parse_final():
    obj = {"type": "final", "summary": "Done", "changes": []}
    a = parse_action(obj)
    assert isinstance(a, FinalAction)
    assert a.summary == "Done"


def test_invalid_tool_call_raises():
    obj = {"type": "tool_call", "name": "", "args": {}}
    with pytest.raises(ActionParseError):
        parse_action(obj)


def test_invalid_type_raises():
    with pytest.raises(ActionParseError):
        parse_action("not a dict")
