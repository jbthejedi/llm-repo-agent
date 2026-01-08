"""Tests for prefs/schema.py - PreferencePair, PreferenceMeta, format_together_jsonl."""

import json

from llm_repo_agent.prefs.schema import (
    PreferencePair,
    PreferenceMeta,
    format_together_jsonl,
)


def test_preference_meta_basic():
    """Test PreferenceMeta creation and basic attributes."""
    meta = PreferenceMeta(
        task_id="fix_quicksort",
        suite="my_suite",
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        temperature=0.7,
        seed=42,
        scores={"preferred": 1.0, "non_preferred": 0.0},
        tests_ok={"preferred": True, "non_preferred": False},
        trace_ids={"preferred": "run_abc", "non_preferred": "run_xyz"},
        rollout_counts={"total": 4},
    )
    assert meta.task_id == "fix_quicksort"
    assert meta.suite == "my_suite"
    assert meta.model == "Qwen/Qwen2.5-72B-Instruct-Turbo"
    assert meta.temperature == 0.7
    assert meta.seed == 42
    assert meta.scores["preferred"] == 1.0
    assert meta.scores["non_preferred"] == 0.0
    assert meta.tests_ok["preferred"] is True
    assert meta.tests_ok["non_preferred"] is False


def test_preference_meta_to_dict():
    """Test PreferenceMeta serialization to dict."""
    meta = PreferenceMeta(
        task_id="task1",
        suite="suite1",
        model="model1",
        temperature=0.5,
        seed=123,
        scores={"preferred": 1.0, "non_preferred": 0.5},
        tests_ok={"preferred": True, "non_preferred": True},
        trace_ids={"preferred": "run1", "non_preferred": "run2"},
        rollout_counts={"total": 2},
    )
    d = meta.to_dict()
    assert d["task_id"] == "task1"
    assert d["suite"] == "suite1"
    assert d["model"] == "model1"
    assert d["temperature"] == 0.5
    assert d["seed"] == 123
    assert d["scores"] == {"preferred": 1.0, "non_preferred": 0.5}
    assert d["tests_ok"] == {"preferred": True, "non_preferred": True}
    assert d["trace_ids"] == {"preferred": "run1", "non_preferred": "run2"}
    assert d["rollout_counts"] == {"total": 2}


def test_preference_meta_to_jsonl():
    """Test PreferenceMeta serialization to JSONL."""
    meta = PreferenceMeta(
        task_id="task1",
        suite="suite1",
        model="model1",
        temperature=0.5,
        seed=123,
        scores={"preferred": 1.0, "non_preferred": 0.0},
        tests_ok={"preferred": True, "non_preferred": False},
        trace_ids={"preferred": "run1", "non_preferred": "run2"},
        rollout_counts={"total": 4},
    )
    jsonl = meta.to_jsonl()
    parsed = json.loads(jsonl)
    assert parsed["task_id"] == "task1"
    assert parsed["scores"]["preferred"] == 1.0


def test_preference_pair_basic():
    """Test PreferencePair creation and basic attributes."""
    pair = PreferencePair(
        input_messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "GOAL:\nFix the bug."},
        ],
        preferred_content='{"type":"final","summary":"Fixed it."}',
        non_preferred_content='{"type":"final","summary":"Could not fix."}',
    )
    assert len(pair.input_messages) == 2
    assert pair.input_messages[0]["role"] == "system"
    assert pair.input_messages[1]["role"] == "user"
    assert "Fixed it" in pair.preferred_content
    assert "Could not fix" in pair.non_preferred_content


def test_preference_pair_to_together_format():
    """Test PreferencePair conversion to Together's expected format."""
    pair = PreferencePair(
        input_messages=[
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User goal."},
        ],
        preferred_content="Preferred response.",
        non_preferred_content="Non-preferred response.",
    )
    together_format = pair.to_together_format()

    # Check input structure
    assert "input" in together_format
    assert "messages" in together_format["input"]
    assert len(together_format["input"]["messages"]) == 2
    assert together_format["input"]["messages"][0]["role"] == "system"
    assert together_format["input"]["messages"][1]["role"] == "user"

    # Check preferred_output structure
    assert "preferred_output" in together_format
    assert isinstance(together_format["preferred_output"], list)
    assert len(together_format["preferred_output"]) == 1
    assert together_format["preferred_output"][0]["role"] == "assistant"
    assert together_format["preferred_output"][0]["content"] == "Preferred response."

    # Check non_preferred_output structure
    assert "non_preferred_output" in together_format
    assert isinstance(together_format["non_preferred_output"], list)
    assert len(together_format["non_preferred_output"]) == 1
    assert together_format["non_preferred_output"][0]["role"] == "assistant"
    assert together_format["non_preferred_output"][0]["content"] == "Non-preferred response."


def test_preference_pair_to_jsonl():
    """Test PreferencePair serialization to JSONL."""
    pair = PreferencePair(
        input_messages=[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Goal."},
        ],
        preferred_content="Good.",
        non_preferred_content="Bad.",
    )
    jsonl = pair.to_jsonl()
    parsed = json.loads(jsonl)

    assert parsed["input"]["messages"][0]["content"] == "Sys."
    assert parsed["preferred_output"][0]["content"] == "Good."
    assert parsed["non_preferred_output"][0]["content"] == "Bad."


def test_format_together_jsonl():
    """Test format_together_jsonl convenience function."""
    pair = format_together_jsonl(
        system_prompt="You are a repo-fixing agent.",
        user_goal="GOAL:\nFix quicksort.",
        preferred_content='{"type":"final","summary":"Fixed."}',
        non_preferred_content='{"type":"final","summary":"Failed."}',
    )

    assert isinstance(pair, PreferencePair)
    assert len(pair.input_messages) == 2
    assert pair.input_messages[0]["role"] == "system"
    assert pair.input_messages[0]["content"] == "You are a repo-fixing agent."
    assert pair.input_messages[1]["role"] == "user"
    assert pair.input_messages[1]["content"] == "GOAL:\nFix quicksort."
    assert pair.preferred_content == '{"type":"final","summary":"Fixed."}'
    assert pair.non_preferred_content == '{"type":"final","summary":"Failed."}'


def test_format_together_jsonl_produces_valid_together_format():
    """Test that format_together_jsonl produces valid Together format."""
    pair = format_together_jsonl(
        system_prompt="System.",
        user_goal="Goal.",
        preferred_content="Preferred.",
        non_preferred_content="Non-preferred.",
    )
    together_format = pair.to_together_format()

    # Verify the exact structure Together expects
    assert set(together_format.keys()) == {"input", "preferred_output", "non_preferred_output"}
    assert set(together_format["input"].keys()) == {"messages"}
    assert all(
        set(msg.keys()) == {"role", "content"}
        for msg in together_format["input"]["messages"]
    )
    assert all(
        set(msg.keys()) == {"role", "content"}
        for msg in together_format["preferred_output"]
    )
    assert all(
        set(msg.keys()) == {"role", "content"}
        for msg in together_format["non_preferred_output"]
    )
