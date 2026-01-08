"""Scoring logic for rollouts based on test outcomes and run metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_repo_agent.eval.runner import TaskResult


@dataclass
class RolloutScore:
    """Score for a single rollout.

    Primary signal: tests pass (1.0) or fail (0.0)
    Tie-breakers (lower is better):
      - steps: number of agent iterations
      - tool_calls: number of tool calls
      - files_touched: number of files modified
    """
    primary: float  # 1.0 if tests passed, 0.0 otherwise
    steps: int
    tool_calls: int
    files_touched: int

    def __lt__(self, other: "RolloutScore") -> bool:
        """Compare scores: higher primary is better, then lower tie-breakers."""
        # Higher primary score is better
        if self.primary != other.primary:
            return self.primary < other.primary
        # Lower steps is better (fewer iterations)
        if self.steps != other.steps:
            return self.steps > other.steps
        # Lower tool calls is better
        if self.tool_calls != other.tool_calls:
            return self.tool_calls > other.tool_calls
        # Lower files touched is better
        return self.files_touched > other.files_touched

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RolloutScore):
            return NotImplemented
        return (
            self.primary == other.primary
            and self.steps == other.steps
            and self.tool_calls == other.tool_calls
            and self.files_touched == other.files_touched
        )


def score_rollout(result: TaskResult) -> RolloutScore:
    """Compute a score from a TaskResult.

    Args:
        result: The TaskResult from running a single rollout.

    Returns:
        RolloutScore with primary signal and tie-breaker metrics.
    """
    # Primary signal: tests pass = 1.0, fail = 0.0
    if result.success is True:
        primary = 1.0
    elif result.success is False:
        primary = 0.0
    else:
        # No test result (success is None) - treat as failure
        primary = 0.0

    return RolloutScore(
        primary=primary,
        steps=result.steps,
        tool_calls=result.tool_calls,
        files_touched=len(result.files_touched),
    )
