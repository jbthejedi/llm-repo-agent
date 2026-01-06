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
