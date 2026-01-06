"""Task specification and suite loading for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
  """Specification for a single evaluation task.

    Attributes:
        task_id: Unique identifier for this task.
        repo: Path to the repository to operate on.
        goal: The goal/instruction for the agent.
        test_cmd: Command to run tests (space-separated string or list).
        metadata: Optional additional metadata (e.g., difficulty, category, expected_files).
    """

  task_id: str
  repo: str
  goal: str
  test_cmd: str = ""
  metadata: Dict[str, Any] = field(default_factory=dict)

  def test_cmd_list(self) -> List[str]:
    """Return test command as a list of arguments."""
    if not self.test_cmd.strip():
      return []
    return self.test_cmd.split()

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
    return cls(
        task_id=d["task_id"],
        repo=d["repo"],
        goal=d["goal"],
        test_cmd=d.get("test_cmd", ""),
        metadata=d.get("metadata", {}),
    )


@dataclass
class EvalSuite:
  """A collection of tasks to evaluate.

    Attributes:
        name: Name of the evaluation suite.
        description: Optional description.
        tasks: List of TaskSpec objects.
        defaults: Default values applied to all tasks (e.g., default test_cmd).
    """

  name: str
  tasks: List[TaskSpec]
  description: str = ""
  defaults: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "name": self.name,
        "description": self.description,
        "defaults": self.defaults,
        "tasks": [t.to_dict() for t in self.tasks],
    }

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "EvalSuite":
    defaults = d.get("defaults", {})
    tasks = []
    for t in d.get("tasks", []):
      # Apply defaults to each task
      task_data = {**defaults, **t}
      tasks.append(TaskSpec.from_dict(task_data))
    return cls(
        name=d.get("name", "unnamed"),
        description=d.get("description", ""),
        defaults=defaults,
        tasks=tasks,
    )


def load_suite(path: Path) -> EvalSuite:
  """Load an evaluation suite from a JSON file.

    The JSON format:
    {
        "name": "my_suite",
        "description": "Optional description",
        "defaults": {
            "repo": "/path/to/default/repo",
            "test_cmd": "python -m pytest -q"
        },
        "tasks": [
            {
                "task_id": "fix_quicksort",
                "goal": "Fix the quicksort implementation so tests pass",
                "metadata": {"difficulty": "easy", "category": "sorting"}
            },
            ...
        ]
    }
    """
  path = Path(path).expanduser().resolve()
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  return EvalSuite.from_dict(data)


def save_suite(suite: EvalSuite, path: Path) -> None:
  """Save an evaluation suite to a JSON file."""
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(suite.to_dict(), f, indent=2)
