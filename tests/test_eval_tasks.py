"""Tests for eval/tasks.py - TaskSpec, EvalSuite, load/save suite."""

import json
from pathlib import Path

import llm_repo_agent.eval.tasks as eval_tasks


def test_task_spec_basic():
    """Test TaskSpec creation and basic attributes."""
    task = eval_tasks.TaskSpec(
        task_id="test_task",
        repo="/path/to/repo",
        goal="Fix the bug",
        test_cmd="pytest -q",
        metadata={"category": "bugfix"},
    )
    assert task.task_id == "test_task"
    assert task.repo == "/path/to/repo"
    assert task.goal == "Fix the bug"
    assert task.test_cmd == "pytest -q"
    assert task.metadata == {"category": "bugfix"}


def test_task_spec_defaults():
    """Test TaskSpec with default values."""
    task = eval_tasks.TaskSpec(
        task_id="minimal",
        repo="/repo",
        goal="Do something",
    )
    assert task.test_cmd == ""
    assert task.metadata == {}


def test_task_spec_test_cmd_list():
    """Test test_cmd_list() splits command correctly."""
    task = eval_tasks.TaskSpec(
        task_id="t1",
        repo="/repo",
        goal="goal",
        test_cmd="python -m pytest -q tests/",
    )
    assert task.test_cmd_list() == ["python", "-m", "pytest", "-q", "tests/"]


def test_task_spec_test_cmd_list_empty():
    """Test test_cmd_list() returns empty list for empty command."""
    task = eval_tasks.TaskSpec(
        task_id="t1",
        repo="/repo",
        goal="goal",
        test_cmd="",
    )
    assert task.test_cmd_list() == []

    task_whitespace = eval_tasks.TaskSpec(
        task_id="t2",
        repo="/repo",
        goal="goal",
        test_cmd="   ",
    )
    assert task_whitespace.test_cmd_list() == []


def test_task_spec_to_dict():
    """Test TaskSpec serialization to dict."""
    task = eval_tasks.TaskSpec(
        task_id="t1",
        repo="/repo",
        goal="goal",
        test_cmd="pytest",
        metadata={"key": "value"},
    )
    d = task.to_dict()
    assert d["task_id"] == "t1"
    assert d["repo"] == "/repo"
    assert d["goal"] == "goal"
    assert d["test_cmd"] == "pytest"
    assert d["metadata"] == {"key": "value"}


def test_task_spec_from_dict():
    """Test TaskSpec deserialization from dict."""
    d = {
        "task_id": "t1",
        "repo": "/repo",
        "goal": "goal",
        "test_cmd": "pytest",
        "metadata": {"category": "test"},
    }
    task = eval_tasks.TaskSpec.from_dict(d)
    assert task.task_id == "t1"
    assert task.repo == "/repo"
    assert task.goal == "goal"
    assert task.test_cmd == "pytest"
    assert task.metadata == {"category": "test"}


def test_task_spec_from_dict_missing_optional():
    """Test TaskSpec.from_dict with missing optional fields."""
    d = {
        "task_id": "t1",
        "repo": "/repo",
        "goal": "goal",
    }
    task = eval_tasks.TaskSpec.from_dict(d)
    assert task.test_cmd == ""
    assert task.metadata == {}


def test_eval_suite_basic():
    """Test EvalSuite creation and basic attributes."""
    tasks = [
        eval_tasks.TaskSpec(task_id="t1", repo="/r", goal="g1"),
        eval_tasks.TaskSpec(task_id="t2", repo="/r", goal="g2"),
    ]
    suite = eval_tasks.EvalSuite(
        name="test_suite",
        tasks=tasks,
        description="A test suite",
        defaults={"repo": "/default"},
    )
    assert suite.name == "test_suite"
    assert suite.description == "A test suite"
    assert len(suite.tasks) == 2
    assert suite.defaults == {"repo": "/default"}


def test_eval_suite_to_dict():
    """Test EvalSuite serialization."""
    tasks = [eval_tasks.TaskSpec(task_id="t1", repo="/r", goal="g1")]
    suite = eval_tasks.EvalSuite(
        name="suite",
        tasks=tasks,
        description="desc",
        defaults={"test_cmd": "pytest"},
    )
    d = suite.to_dict()
    assert d["name"] == "suite"
    assert d["description"] == "desc"
    assert d["defaults"] == {"test_cmd": "pytest"}
    assert len(d["tasks"]) == 1
    assert d["tasks"][0]["task_id"] == "t1"


def test_eval_suite_from_dict_with_defaults():
    """Test EvalSuite.from_dict applies defaults to tasks."""
    d = {
        "name": "suite",
        "description": "test",
        "defaults": {
            "repo": "/default/repo",
            "test_cmd": "pytest -q",
        },
        "tasks": [
            {"task_id": "t1", "goal": "goal1"},
            {"task_id": "t2", "goal": "goal2", "repo": "/custom/repo"},
        ],
    }
    suite = eval_tasks.EvalSuite.from_dict(d)
    assert suite.name == "suite"
    assert len(suite.tasks) == 2

    # Task 1 should inherit defaults
    assert suite.tasks[0].repo == "/default/repo"
    assert suite.tasks[0].test_cmd == "pytest -q"

    # Task 2 overrides repo but inherits test_cmd
    assert suite.tasks[1].repo == "/custom/repo"
    assert suite.tasks[1].test_cmd == "pytest -q"


def test_load_suite(tmp_path):
    """Test loading a suite from JSON file."""
    suite_data = {
        "name": "loaded_suite",
        "description": "From file",
        "defaults": {"repo": "/default"},
        "tasks": [
            {"task_id": "t1", "goal": "goal1"},
        ],
    }
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps(suite_data), encoding="utf-8")

    suite = eval_tasks.load_suite(suite_file)
    assert suite.name == "loaded_suite"
    assert suite.description == "From file"
    assert len(suite.tasks) == 1
    assert suite.tasks[0].task_id == "t1"
    assert suite.tasks[0].repo == "/default"


def test_save_suite(tmp_path):
    """Test saving a suite to JSON file."""
    tasks = [eval_tasks.TaskSpec(task_id="t1", repo="/r", goal="g")]
    suite = eval_tasks.EvalSuite(name="save_test", tasks=tasks)

    out_path = tmp_path / "subdir" / "suite.json"
    eval_tasks.save_suite(suite, out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["name"] == "save_test"
    assert len(loaded["tasks"]) == 1


def test_roundtrip_suite(tmp_path):
    """Test that save then load preserves suite data."""
    original = eval_tasks.EvalSuite(
        name="roundtrip",
        description="Testing roundtrip",
        defaults={"test_cmd": "make test"},
        tasks=[
            eval_tasks.TaskSpec(
                task_id="task1",
                repo="/repo1",
                goal="Fix bug",
                test_cmd="pytest",
                metadata={"difficulty": "easy"},
            ),
        ],
    )

    path = tmp_path / "roundtrip.json"
    eval_tasks.save_suite(original, path)
    loaded = eval_tasks.load_suite(path)

    assert loaded.name == original.name
    assert loaded.description == original.description
    assert len(loaded.tasks) == len(original.tasks)
    assert loaded.tasks[0].task_id == original.tasks[0].task_id
    assert loaded.tasks[0].metadata == original.tasks[0].metadata
