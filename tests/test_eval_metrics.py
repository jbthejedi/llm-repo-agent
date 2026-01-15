"""Tests for eval/metrics.py - compute_metrics, EvalMetrics, format_metrics_summary."""

import llm_repo_agent.eval.metrics as eval_metrics
import llm_repo_agent.eval.runner as eval_runner


def _make_result(
    task_id: str,
    success: bool = None,
    error: str = None,
    steps: int = 1,
    tool_calls: int = 1,
    duration_s: float = 1.0,
    reflection_count: int = 0,
    parse_errors: int = 0,
    test_runs: int = 0,
    valid_tool_actions: int = 0,
    metadata: dict = None,
) -> eval_runner.TaskResult:
    """Helper to create TaskResult objects for testing."""
    return eval_runner.TaskResult(
        task_id=task_id,
        run_id=f"run_{task_id}",
        success=success,
        error=error,
        steps=steps,
        tool_calls=tool_calls,
        duration_s=duration_s,
        reflection_count=reflection_count,
        parse_errors=parse_errors,
        test_runs=test_runs,
        valid_tool_actions=valid_tool_actions,
        metadata=metadata or {},
    )


def test_compute_metrics_empty():
    """Test compute_metrics with empty results list."""
    metrics = eval_metrics.compute_metrics([])
    assert metrics.total_tasks == 0
    assert metrics.passed == 0
    assert metrics.failed == 0
    assert metrics.success_rate == 0.0


def test_compute_metrics_all_passed():
    """Test compute_metrics when all tasks pass."""
    results = [
        _make_result("t1", success=True, steps=2, tool_calls=3, duration_s=1.0),
        _make_result("t2", success=True, steps=4, tool_calls=5, duration_s=2.0),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_tasks == 2
    assert metrics.passed == 2
    assert metrics.failed == 0
    assert metrics.errored == 0
    assert metrics.success_rate == 1.0
    assert metrics.avg_steps == 3.0
    assert metrics.avg_tool_calls == 4.0
    assert metrics.avg_duration_s == 1.5
    assert metrics.total_duration_s == 3.0


def test_compute_metrics_all_failed():
    """Test compute_metrics when all tasks fail."""
    results = [
        _make_result("t1", success=False),
        _make_result("t2", success=False),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_tasks == 2
    assert metrics.passed == 0
    assert metrics.failed == 2
    assert metrics.success_rate == 0.0


def test_compute_metrics_mixed():
    """Test compute_metrics with mixed pass/fail results."""
    results = [
        _make_result("t1", success=True),
        _make_result("t2", success=False),
        _make_result("t3", success=True),
        _make_result("t4", success=False),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_tasks == 4
    assert metrics.passed == 2
    assert metrics.failed == 2
    assert metrics.success_rate == 0.5


def test_compute_metrics_errored():
    """Test compute_metrics counts errors correctly."""
    results = [
        _make_result("t1", success=True),
        _make_result("t2", error="Something broke"),
        _make_result("t3", success=False),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_tasks == 3
    assert metrics.passed == 1
    assert metrics.failed == 1
    assert metrics.errored == 1
    # Success rate should exclude errored
    assert metrics.success_rate == 0.5


def test_compute_metrics_no_tests():
    """Test compute_metrics with no test results (success=None)."""
    results = [
        _make_result("t1", success=None),
        _make_result("t2", success=True),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.no_tests == 1
    assert metrics.passed == 1
    assert metrics.success_rate == 1.0  # Only counts tested tasks


def test_compute_metrics_reflection_and_errors():
    """Test compute_metrics aggregates reflection and parse error counts."""
    results = [
        _make_result("t1", success=True, reflection_count=2, parse_errors=1, test_runs=3),
        _make_result("t2", success=True, reflection_count=1, parse_errors=0, test_runs=2),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_reflections == 3
    assert metrics.avg_reflections == 1.5
    assert metrics.total_parse_errors == 1
    assert metrics.avg_parse_errors == 0.5
    assert metrics.avg_test_runs == 2.5


def test_compute_metrics_by_category():
    """Test compute_metrics groups results by category."""
    results = [
        _make_result("t1", success=True, metadata={"category": "sorting"}),
        _make_result("t2", success=False, metadata={"category": "sorting"}),
        _make_result("t3", success=True, metadata={"category": "math"}),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert "sorting" in metrics.by_category
    assert "math" in metrics.by_category

    sorting = metrics.by_category["sorting"]
    assert sorting.total_tasks == 2
    assert sorting.passed == 1
    assert sorting.failed == 1
    assert sorting.success_rate == 0.5

    math = metrics.by_category["math"]
    assert math.total_tasks == 1
    assert math.passed == 1
    assert math.success_rate == 1.0


def test_compute_metrics_uncategorized():
    """Test tasks without category go to 'uncategorized'."""
    results = [
        _make_result("t1", success=True),  # No category
        _make_result("t2", success=True, metadata={}),  # Empty metadata
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert "uncategorized" in metrics.by_category
    assert metrics.by_category["uncategorized"].total_tasks == 2


def test_eval_metrics_to_dict():
    """Test EvalMetrics serialization to dict."""
    metrics = eval_metrics.EvalMetrics(
        total_tasks=5,
        passed=3,
        failed=2,
        success_rate=0.6,
    )
    d = metrics.to_dict()
    assert d["total_tasks"] == 5
    assert d["passed"] == 3
    assert d["success_rate"] == 0.6


def test_eval_metrics_to_dict_with_category():
    """Test EvalMetrics.to_dict handles nested by_category."""
    inner = eval_metrics.EvalMetrics(total_tasks=1, passed=1)
    metrics = eval_metrics.EvalMetrics(
        total_tasks=1,
        by_category={"test": inner},
    )
    d = metrics.to_dict()
    assert "test" in d["by_category"]
    assert d["by_category"]["test"]["total_tasks"] == 1


def test_format_metrics_summary():
    """Test format_metrics_summary produces readable output."""
    metrics = eval_metrics.EvalMetrics(
        total_tasks=10,
        passed=7,
        failed=2,
        errored=1,
        no_tests=0,
        success_rate=0.777,
        avg_steps=5.5,
        avg_tool_calls=8.2,
        avg_reflections=1.3,
        avg_test_runs=2.1,
        total_parse_errors=3,
        avg_duration_s=12.5,
        total_duration_s=125.0,
    )
    summary = eval_metrics.format_metrics_summary(metrics)

    assert "EVALUATION SUMMARY" in summary
    assert "Total tasks:     10" in summary
    assert "Passed:          7" in summary
    assert "Failed:          2" in summary
    assert "Errored:         1" in summary
    assert "Success rate:    77.7%" in summary
    assert "Avg steps:       5.5" in summary
    assert "Avg reflections: 1.3" in summary
    assert "Parse errors:    3" in summary


def test_format_metrics_summary_with_categories():
    """Test format_metrics_summary includes category breakdown."""
    cat1 = eval_metrics.EvalMetrics(total_tasks=2, passed=2, success_rate=1.0)
    cat2 = eval_metrics.EvalMetrics(total_tasks=3, passed=1, success_rate=0.333)
    metrics = eval_metrics.EvalMetrics(
        total_tasks=5,
        passed=3,
        by_category={"easy": cat1, "hard": cat2},
    )
    summary = eval_metrics.format_metrics_summary(metrics)

    assert "BY CATEGORY:" in summary
    assert "easy: 2/2 (100%)" in summary
    assert "hard: 1/3 (33%)" in summary


# ============================================================
# Tool Call Instruction Following Tests
# ============================================================


def test_compute_metrics_tool_parse_success_rate_all_valid():
    """Test tool_parse_success_rate when all tool calls are valid."""
    results = [
        _make_result("t1", success=True, valid_tool_actions=5, parse_errors=0),
        _make_result("t2", success=True, valid_tool_actions=3, parse_errors=0),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_valid_tool_actions == 8
    assert metrics.total_parse_errors == 0
    assert metrics.tool_parse_success_rate == 1.0


def test_compute_metrics_tool_parse_success_rate_mixed():
    """Test tool_parse_success_rate with mixed valid/invalid tool calls."""
    results = [
        _make_result("t1", success=True, valid_tool_actions=8, parse_errors=2),
        _make_result("t2", success=False, valid_tool_actions=6, parse_errors=4),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_valid_tool_actions == 14
    assert metrics.total_parse_errors == 6
    # 14 / (14 + 6) = 14/20 = 0.7
    assert metrics.tool_parse_success_rate == 0.7


def test_compute_metrics_tool_parse_success_rate_all_errors():
    """Test tool_parse_success_rate when all tool calls fail to parse."""
    results = [
        _make_result("t1", success=False, valid_tool_actions=0, parse_errors=5),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_valid_tool_actions == 0
    assert metrics.total_parse_errors == 5
    assert metrics.tool_parse_success_rate == 0.0


def test_compute_metrics_tool_parse_success_rate_no_tool_calls():
    """Test tool_parse_success_rate when there are no tool call attempts."""
    results = [
        _make_result("t1", success=True, valid_tool_actions=0, parse_errors=0),
    ]
    metrics = eval_metrics.compute_metrics(results)

    assert metrics.total_valid_tool_actions == 0
    assert metrics.total_parse_errors == 0
    # No attempts = 0% (default)
    assert metrics.tool_parse_success_rate == 0.0


def test_format_metrics_summary_includes_tool_instruction_following():
    """Test format_metrics_summary includes tool call instruction following section."""
    metrics = eval_metrics.EvalMetrics(
        total_tasks=2,
        passed=1,
        failed=1,
        success_rate=0.5,
        total_valid_tool_actions=18,
        total_parse_errors=2,
        tool_parse_success_rate=0.9,
    )
    summary = eval_metrics.format_metrics_summary(metrics)

    assert "TOOL CALL INSTRUCTION FOLLOWING:" in summary
    assert "Valid tool actions:     18" in summary
    assert "Parse errors:           2" in summary
    assert "Tool parse success:     90.0%" in summary


def test_task_result_includes_valid_tool_actions():
    """Test TaskResult has valid_tool_actions field."""
    result = eval_runner.TaskResult(
        task_id="test",
        run_id="run_test",
        valid_tool_actions=10,
    )
    assert result.valid_tool_actions == 10
    d = result.to_dict()
    assert d["valid_tool_actions"] == 10


# ============================================================
# Rollout Metrics Tests
# ============================================================


def test_rollout_results_get_task_pass_rate():
    """Test RolloutResults.get_task_pass_rate calculates correctly."""
    task_results = {
        "task1": [
            _make_result("task1", success=True),
            _make_result("task1", success=True),
            _make_result("task1", success=False),
        ],
        "task2": [
            _make_result("task2", success=False),
            _make_result("task2", success=False),
        ],
    }
    rollout_results = eval_runner.RolloutResults(
        task_results=task_results,
        rollouts_per_task=3,
        total_tasks=2,
    )

    # task1: 2/3 = 66.7%
    assert abs(rollout_results.get_task_pass_rate("task1") - 0.667) < 0.01
    # task2: 0/2 = 0%
    assert rollout_results.get_task_pass_rate("task2") == 0.0
    # unknown task
    assert rollout_results.get_task_pass_rate("unknown") == 0.0


def test_rollout_results_get_task_summary():
    """Test RolloutResults.get_task_summary returns correct breakdown."""
    task_results = {
        "task1": [
            _make_result("task1", success=True),
            _make_result("task1", success=False),
            _make_result("task1", error="boom"),
        ],
    }
    rollout_results = eval_runner.RolloutResults(
        task_results=task_results,
        rollouts_per_task=3,
        total_tasks=1,
    )

    summary = rollout_results.get_task_summary("task1")
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["errored"] == 1
    assert summary["total"] == 3
    assert summary["pass_rate"] == 0.5  # 1 passed / 2 tested


def test_compute_metrics_with_rollouts_basic():
    """Test compute_metrics_with_rollouts computes rollout-specific metrics."""
    task_results = {
        "task1": [
            _make_result("task1", success=True, steps=2, tool_calls=3),
            _make_result("task1", success=True, steps=3, tool_calls=4),
        ],
        "task2": [
            _make_result("task2", success=True, steps=1, tool_calls=2),
            _make_result("task2", success=False, steps=2, tool_calls=3),
        ],
    }
    rollout_results = eval_runner.RolloutResults(
        task_results=task_results,
        rollouts_per_task=2,
        total_tasks=2,
    )

    metrics = eval_metrics.compute_metrics_with_rollouts(rollout_results)

    assert metrics.total_tasks == 2
    assert metrics.rollouts_per_task == 2
    assert metrics.total_attempts == 4
    assert metrics.passed == 3
    assert metrics.failed == 1
    assert metrics.success_rate == 0.75  # 3/4


def test_compute_metrics_with_rollouts_consistency():
    """Test compute_metrics_with_rollouts tracks consistency correctly."""
    task_results = {
        "always_pass": [
            _make_result("always_pass", success=True),
            _make_result("always_pass", success=True),
            _make_result("always_pass", success=True),
        ],
        "always_fail": [
            _make_result("always_fail", success=False),
            _make_result("always_fail", success=False),
        ],
        "mixed": [
            _make_result("mixed", success=True),
            _make_result("mixed", success=False),
            _make_result("mixed", success=True),
        ],
    }
    rollout_results = eval_runner.RolloutResults(
        task_results=task_results,
        rollouts_per_task=3,
        total_tasks=3,
    )

    metrics = eval_metrics.compute_metrics_with_rollouts(rollout_results)

    assert metrics.consistent_pass == 1  # always_pass
    assert metrics.consistent_fail == 1  # always_fail
    assert metrics.inconsistent == 1  # mixed


def test_compute_metrics_with_rollouts_avg_task_pass_rate():
    """Test compute_metrics_with_rollouts calculates avg_task_pass_rate correctly."""
    task_results = {
        # 100% pass rate
        "task1": [_make_result("task1", success=True), _make_result("task1", success=True)],
        # 50% pass rate
        "task2": [_make_result("task2", success=True), _make_result("task2", success=False)],
        # 0% pass rate
        "task3": [_make_result("task3", success=False), _make_result("task3", success=False)],
    }
    rollout_results = eval_runner.RolloutResults(
        task_results=task_results,
        rollouts_per_task=2,
        total_tasks=3,
    )

    metrics = eval_metrics.compute_metrics_with_rollouts(rollout_results)

    # Average of 100%, 50%, 0% = 50%
    assert abs(metrics.avg_task_pass_rate - 0.5) < 0.01


def test_format_metrics_summary_with_rollouts():
    """Test format_metrics_summary shows rollout-specific info."""
    metrics = eval_metrics.EvalMetrics(
        total_tasks=3,
        passed=5,
        failed=1,
        rollouts_per_task=2,
        total_attempts=6,
        avg_task_pass_rate=0.833,
        consistent_pass=2,
        consistent_fail=0,
        inconsistent=1,
        per_task_results={
            "task1": {"passed": 2, "total": 2, "pass_rate": 1.0},
            "task2": {"passed": 2, "total": 2, "pass_rate": 1.0},
            "task3": {"passed": 1, "total": 2, "pass_rate": 0.5},
        },
    )

    summary = eval_metrics.format_metrics_summary(metrics)

    assert "Tasks:           3" in summary
    assert "Rollouts/task:   2" in summary
    assert "Total attempts:  6" in summary
    assert "Avg pass rate: 83.3%" in summary
    assert "Always pass:   2/3" in summary
    assert "Mixed:         1/3" in summary
    assert "PER-TASK RESULTS:" in summary
    assert "task1: 2/2 (100%)" in summary
    assert "task3: 1/2 (50%)" in summary


def test_format_metrics_summary_single_rollout_unchanged():
    """Test format_metrics_summary still works for single rollout (no rollout info)."""
    metrics = eval_metrics.EvalMetrics(
        total_tasks=5,
        passed=3,
        failed=2,
        success_rate=0.6,
        rollouts_per_task=1,  # Single rollout
    )

    summary = eval_metrics.format_metrics_summary(metrics)

    # Should use original format without rollout info
    assert "Total tasks:     5" in summary
    assert "Passed:          3" in summary
    assert "Failed:          2" in summary
    assert "Success rate:    60.0%" in summary
    # Should NOT have rollout-specific sections
    assert "Rollouts/task" not in summary
    assert "Total attempts" not in summary
    assert "Avg pass rate" not in summary
