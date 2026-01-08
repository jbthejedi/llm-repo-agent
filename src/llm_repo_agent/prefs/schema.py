"""Schema helpers for Together DPO preference data format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PreferenceMeta:
    """Metadata for a preference pair (stored in separate file for debugging)."""
    task_id: str
    suite: str
    model: str
    temperature: float
    seed: int
    scores: Dict[str, float]  # {"preferred": 1.0, "non_preferred": 0.0}
    tests_ok: Dict[str, bool]  # {"preferred": True, "non_preferred": False}
    trace_ids: Dict[str, str]  # {"preferred": "run_abc", "non_preferred": "run_xyz"}
    rollout_counts: Dict[str, int]  # {"total": 4}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class PreferencePair:
    """A preference pair in Together's DPO format.

    Together expects:
    {
      "input": {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."}
        ]
      },
      "preferred_output": [
        {"role": "assistant", "content": "..."}
      ],
      "non_preferred_output": [
        {"role": "assistant", "content": "..."}
      ]
    }
    """
    input_messages: List[Dict[str, str]]
    preferred_content: str
    non_preferred_content: str

    def to_together_format(self) -> Dict[str, Any]:
        """Convert to Together's expected format."""
        return {
            "input": {
                "messages": self.input_messages,
            },
            "preferred_output": [
                {"role": "assistant", "content": self.preferred_content}
            ],
            "non_preferred_output": [
                {"role": "assistant", "content": self.non_preferred_content}
            ],
        }

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.to_together_format())


def format_together_jsonl(
    system_prompt: str,
    user_goal: str,
    preferred_content: str,
    non_preferred_content: str,
) -> PreferencePair:
    """Create a PreferencePair from components."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_goal},
    ]
    return PreferencePair(
        input_messages=messages,
        preferred_content=preferred_content,
        non_preferred_content=non_preferred_content,
    )