"""Metrics computation for evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .runner import TaskResult


# Keywords that indicate a parse/JSON error in exception messages
_PARSE_ERROR_KEYWORDS = [
    "parse", "json", "valid type", "malformed", "decode", "unexpected token",
    "expecting", "unterminated", "invalid syntax", "failed to produce",
]


def _is_parse_error(error_msg: str) -> bool:
  """Check if an error message indicates a JSON/parse error."""
  if not error_msg:
    return False
  error_lower = error_msg.lower()
  return any(kw in error_lower for kw in _PARSE_ERROR_KEYWORDS)


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
  # Tool call instruction following metrics
  total_valid_tool_actions: int = 0
  tool_parse_success_rate: float = 0.0
  by_category: Dict[str, "EvalMetrics"] = field(default_factory=dict)
  # Rollout-specific metrics (only populated when rollouts > 1)
  rollouts_per_task: int = 1
  total_attempts: int = 0  # total_tasks * rollouts_per_task
  avg_task_pass_rate: float = 0.0  # Average of per-task pass rates
  consistent_pass: int = 0  # Tasks where all rollouts passed
  consistent_fail: int = 0  # Tasks where all rollouts failed
  inconsistent: int = 0  # Tasks with mixed results
  per_task_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # task_id -> summary

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    # Convert nested EvalMetrics to dicts
    d["by_category"] = {k: v.to_dict() if isinstance(v, EvalMetrics) else v
                       for k, v in self.by_category.items()}
    # per_task_results is already a dict of dicts, no conversion needed
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
  total_valid_tool_actions = 0

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
    total_valid_tool_actions += getattr(r, 'valid_tool_actions', 0)

    # Count errors that look like parse errors (LLM returned invalid JSON)
    if r.error and _is_parse_error(r.error):
      total_parse_errors += 1

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
  metrics.total_valid_tool_actions = total_valid_tool_actions

  # Compute success rate (excluding errored and no_tests)
  tested = metrics.passed + metrics.failed
  if tested > 0:
    metrics.success_rate = metrics.passed / tested

  # Compute tool call parse success rate (for SFT instruction following evaluation)
  total_tool_attempts = total_valid_tool_actions + total_parse_errors
  if total_tool_attempts > 0:
    metrics.tool_parse_success_rate = total_valid_tool_actions / total_tool_attempts

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
  total_valid_tool_actions = 0

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
    total_valid_tool_actions += getattr(r, 'valid_tool_actions', 0)

    # Count errors that look like parse errors (LLM returned invalid JSON)
    if r.error and _is_parse_error(r.error):
      total_parse_errors += 1

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
  metrics.total_valid_tool_actions = total_valid_tool_actions

  tested = metrics.passed + metrics.failed
  if tested > 0:
    metrics.success_rate = metrics.passed / tested

  # Compute tool call parse success rate
  total_tool_attempts = total_valid_tool_actions + total_parse_errors
  if total_tool_attempts > 0:
    metrics.tool_parse_success_rate = total_valid_tool_actions / total_tool_attempts

  return metrics


def compute_metrics_with_rollouts(rollout_results: "RolloutResults") -> EvalMetrics:
  """Compute aggregate metrics from rollout results.

  Args:
      rollout_results: RolloutResults from running a suite with multiple rollouts.

  Returns:
      EvalMetrics with rollout-specific statistics.
  """
  from .runner import RolloutResults  # Import here to avoid circular import

  all_results = rollout_results.all_results()
  if not all_results:
    return EvalMetrics()

  # Start with base metrics from all individual results
  metrics = compute_metrics(all_results)

  # Add rollout-specific metrics
  metrics.rollouts_per_task = rollout_results.rollouts_per_task
  metrics.total_attempts = len(all_results)
  # Note: total_tasks should be the unique task count, not total attempts
  metrics.total_tasks = rollout_results.total_tasks

  # Compute per-task summaries and consistency metrics
  task_pass_rates = []
  consistent_pass = 0
  consistent_fail = 0
  inconsistent = 0

  for task_id, results in rollout_results.task_results.items():
    summary = rollout_results.get_task_summary(task_id)
    metrics.per_task_results[task_id] = summary

    # Calculate pass rate for this task
    passed = summary["passed"]
    failed = summary["failed"]
    tested = passed + failed

    if tested > 0:
      pass_rate = passed / tested
      task_pass_rates.append(pass_rate)

      # Determine consistency
      if pass_rate == 1.0:
        consistent_pass += 1
      elif pass_rate == 0.0:
        consistent_fail += 1
      else:
        inconsistent += 1

  # Set rollout metrics
  metrics.consistent_pass = consistent_pass
  metrics.consistent_fail = consistent_fail
  metrics.inconsistent = inconsistent

  if task_pass_rates:
    metrics.avg_task_pass_rate = sum(task_pass_rates) / len(task_pass_rates)

  return metrics


def format_metrics_summary(metrics: EvalMetrics) -> str:
  """Format metrics as a human-readable summary string."""
  lines = [
      "=" * 60,
      "EVALUATION SUMMARY",
      "=" * 60,
  ]

  # Show rollout info if running with multiple rollouts
  if metrics.rollouts_per_task > 1:
    lines.extend([
        f"Tasks:           {metrics.total_tasks}",
        f"Rollouts/task:   {metrics.rollouts_per_task}",
        f"Total attempts:  {metrics.total_attempts}",
        "-" * 40,
        "OVERALL (all attempts):",
        f"  Passed:        {metrics.passed}/{metrics.total_attempts}",
        f"  Failed:        {metrics.failed}/{metrics.total_attempts}",
        f"  Errored:       {metrics.errored}",
        f"  Success rate:  {metrics.success_rate:.1%}",
        "-" * 40,
        "PER-TASK AGGREGATION:",
        f"  Avg pass rate: {metrics.avg_task_pass_rate:.1%}",
        f"  Always pass:   {metrics.consistent_pass}/{metrics.total_tasks}",
        f"  Always fail:   {metrics.consistent_fail}/{metrics.total_tasks}",
        f"  Mixed:         {metrics.inconsistent}/{metrics.total_tasks}",
    ])
  else:
    lines.extend([
        f"Total tasks:     {metrics.total_tasks}",
        f"Passed:          {metrics.passed}",
        f"Failed:          {metrics.failed}",
        f"Errored:         {metrics.errored}",
        f"No tests:        {metrics.no_tests}",
        f"Success rate:    {metrics.success_rate:.1%}",
    ])

  lines.extend([
      "-" * 40,
      f"Avg steps:       {metrics.avg_steps:.1f}",
      f"Avg tool calls:  {metrics.avg_tool_calls:.1f}",
      f"Avg reflections: {metrics.avg_reflections:.1f}",
      f"Avg test runs:   {metrics.avg_test_runs:.1f}",
      f"Parse errors:    {metrics.total_parse_errors}",
      f"Avg duration:    {metrics.avg_duration_s:.1f}s",
      f"Total duration:  {metrics.total_duration_s:.1f}s",
      "-" * 40,
      "TOOL CALL INSTRUCTION FOLLOWING:",
      f"  Valid tool actions:     {metrics.total_valid_tool_actions}",
      f"  Parse errors:           {metrics.total_parse_errors}",
      f"  Tool parse success:     {metrics.tool_parse_success_rate:.1%}",
  ])

  # Show per-task breakdown for rollouts
  if metrics.rollouts_per_task > 1 and metrics.per_task_results:
    lines.append("-" * 40)
    lines.append("PER-TASK RESULTS:")
    for task_id, summary in sorted(metrics.per_task_results.items()):
      passed = summary["passed"]
      total = summary["total"]
      pass_rate = summary["pass_rate"]
      lines.append(f"  {task_id}: {passed}/{total} ({pass_rate:.0%})")

  if metrics.by_category:
    lines.append("-" * 40)
    lines.append("BY CATEGORY:")
    for cat, cat_metrics in sorted(metrics.by_category.items()):
      lines.append(f"  {cat}: {cat_metrics.passed}/{cat_metrics.total_tasks} ({cat_metrics.success_rate:.0%})")

  lines.append("=" * 60)
  return "\n".join(lines)
