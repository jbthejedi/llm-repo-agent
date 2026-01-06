"""Metrics computation for evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .runner import TaskResult


@dataclass
class EvalMetrics:
  """Aggregate metrics from an evaluation run.

    Attributes:
        total_tasks: Total number of tasks evaluated.
        passed: Number of tasks that passed (tests succeeded).
        failed: Number of tasks that failed (tests failed).
        errored: Number of tasks that errored (runtime exceptions).
        no_tests: Number of tasks with no test results.
        success_rate: Fraction of passing tasks (passed / (passed + failed)).
        avg_steps: Average number of agent iterations per task.
        avg_tool_calls: Average number of tool calls per task.
        avg_duration_s: Average wall-clock time per task.
        total_duration_s: Total wall-clock time for all tasks.
        avg_reflections: Average number of reflections per task.
        avg_parse_errors: Average number of parse errors per task.
        avg_test_runs: Average number of test executions per task.
        total_reflections: Total reflections across all tasks.
        total_parse_errors: Total parse errors across all tasks.
        by_category: Metrics broken down by category (if metadata includes 'category').
    """

  total_tasks: int = 0
  passed: int = 0
  failed: int = 0
  errored: int = 0
  no_tests: int = 0
  success_rate: float = 0.0
  avg_steps: float = 0.0
  avg_tool_calls: float = 0.0
  avg_duration_s: float = 0.0
  total_duration_s: float = 0.0
  avg_reflections: float = 0.0
  avg_parse_errors: float = 0.0
  avg_test_runs: float = 0.0
  total_reflections: int = 0
  total_parse_errors: int = 0
  by_category: Dict[str, "EvalMetrics"] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    # Convert nested EvalMetrics to dicts
    d["by_category"] = {k: v.to_dict() if isinstance(v, EvalMetrics) else v
                       for k, v in self.by_category.items()}
    return d


def compute_metrics(results: List[TaskResult]) -> EvalMetrics:
  """Compute aggregate metrics from a list of task results.

    Args:
        results: List of TaskResult objects from an evaluation run.

    Returns:
        EvalMetrics with aggregate statistics.
    """
  if not results:
    return EvalMetrics()

  metrics = EvalMetrics(total_tasks=len(results))

  total_steps = 0
  total_tool_calls = 0
  total_duration = 0.0
  total_reflections = 0
  total_parse_errors = 0
  total_test_runs = 0

  # Group by category for per-category metrics
  by_category: Dict[str, List[TaskResult]] = {}

  for r in results:
    # Count outcomes
    if r.error:
      metrics.errored += 1
    elif r.success is True:
      metrics.passed += 1
    elif r.success is False:
      metrics.failed += 1
    else:
      metrics.no_tests += 1

    # Accumulate for averages
    total_steps += r.steps
    total_tool_calls += r.tool_calls
    total_duration += r.duration_s
    total_reflections += r.reflection_count
    total_parse_errors += r.parse_errors
    total_test_runs += r.test_runs

    # Group by category
    category = r.metadata.get("category", "uncategorized")
    if category not in by_category:
      by_category[category] = []
    by_category[category].append(r)

  # Compute averages
  n = len(results)
  metrics.avg_steps = total_steps / n
  metrics.avg_tool_calls = total_tool_calls / n
  metrics.avg_duration_s = total_duration / n
  metrics.total_duration_s = total_duration
  metrics.avg_reflections = total_reflections / n
  metrics.avg_parse_errors = total_parse_errors / n
  metrics.avg_test_runs = total_test_runs / n
  metrics.total_reflections = total_reflections
  metrics.total_parse_errors = total_parse_errors

  # Compute success rate (excluding errored and no_tests)
  tested = metrics.passed + metrics.failed
  if tested > 0:
    metrics.success_rate = metrics.passed / tested

  # Compute per-category metrics (non-recursive to avoid infinite nesting)
  for category, cat_results in by_category.items():
    cat_metrics = _compute_flat_metrics(cat_results)
    metrics.by_category[category] = cat_metrics

  return metrics


def _compute_flat_metrics(results: List[TaskResult]) -> EvalMetrics:
  """Compute metrics without recursive by_category breakdown."""
  if not results:
    return EvalMetrics()

  metrics = EvalMetrics(total_tasks=len(results))

  total_steps = 0
  total_tool_calls = 0
  total_duration = 0.0
  total_reflections = 0
  total_parse_errors = 0
  total_test_runs = 0

  for r in results:
    if r.error:
      metrics.errored += 1
    elif r.success is True:
      metrics.passed += 1
    elif r.success is False:
      metrics.failed += 1
    else:
      metrics.no_tests += 1

    total_steps += r.steps
    total_tool_calls += r.tool_calls
    total_duration += r.duration_s
    total_reflections += r.reflection_count
    total_parse_errors += r.parse_errors
    total_test_runs += r.test_runs

  n = len(results)
  metrics.avg_steps = total_steps / n
  metrics.avg_tool_calls = total_tool_calls / n
  metrics.avg_duration_s = total_duration / n
  metrics.total_duration_s = total_duration
  metrics.avg_reflections = total_reflections / n
  metrics.avg_parse_errors = total_parse_errors / n
  metrics.avg_test_runs = total_test_runs / n
  metrics.total_reflections = total_reflections
  metrics.total_parse_errors = total_parse_errors

  tested = metrics.passed + metrics.failed
  if tested > 0:
    metrics.success_rate = metrics.passed / tested

  return metrics


def format_metrics_summary(metrics: EvalMetrics) -> str:
  """Format metrics as a human-readable summary string."""
  lines = [
      "=" * 60,
      "EVALUATION SUMMARY",
      "=" * 60,
      f"Total tasks:     {metrics.total_tasks}",
      f"Passed:          {metrics.passed}",
      f"Failed:          {metrics.failed}",
      f"Errored:         {metrics.errored}",
      f"No tests:        {metrics.no_tests}",
      f"Success rate:    {metrics.success_rate:.1%}",
      "-" * 40,
      f"Avg steps:       {metrics.avg_steps:.1f}",
      f"Avg tool calls:  {metrics.avg_tool_calls:.1f}",
      f"Avg reflections: {metrics.avg_reflections:.1f}",
      f"Avg test runs:   {metrics.avg_test_runs:.1f}",
      f"Parse errors:    {metrics.total_parse_errors}",
      f"Avg duration:    {metrics.avg_duration_s:.1f}s",
      f"Total duration:  {metrics.total_duration_s:.1f}s",
  ]

  if metrics.by_category:
    lines.append("-" * 40)
    lines.append("BY CATEGORY:")
    for cat, cat_metrics in sorted(metrics.by_category.items()):
      lines.append(f"  {cat}: {cat_metrics.passed}/{cat_metrics.total_tasks} ({cat_metrics.success_rate:.0%})")

  lines.append("=" * 60)
  return "\n".join(lines)
