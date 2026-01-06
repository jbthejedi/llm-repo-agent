"""Tests for eval/report.py - generate_report, write/load_report, compare_reports."""

import json
from pathlib import Path

import llm_repo_agent.eval.report as eval_report
import llm_repo_agent.eval.runner as eval_runner
import llm_repo_agent.eval.metrics as eval_metrics


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


def test_generate_report_basic():
    """Test generate_report creates a report with computed metrics."""
    results = [
        _make_result("t1", success=True, steps=2),
        _make_result("t2", success=False, steps=4),
    ]
    report = eval_report.generate_report(
        suite_name="test_suite",
        results=results,
        config={"model": "gpt-4"},
    )

    assert report.suite_name == "test_suite"
    assert len(report.results) == 2
    assert report.config == {"model": "gpt-4"}
    assert report.metrics.total_tasks == 2
    assert report.metrics.passed == 1
    assert report.metrics.failed == 1
    assert report.timestamp  # Should have a timestamp


def test_generate_report_no_config():
    """Test generate_report with no config provided."""
    results = [_make_result("t1", success=True)]
    report = eval_report.generate_report(
        suite_name="suite",
        results=results,
    )
    assert report.config == {}


def test_eval_report_to_dict():
    """Test EvalReport serialization to dict."""
    results = [_make_result("t1", success=True)]
    report = eval_report.generate_report("suite", results)
    d = report.to_dict()

    assert d["suite_name"] == "suite"
    assert "timestamp" in d
    assert "metrics" in d
    assert "results" in d
    assert len(d["results"]) == 1


def test_write_report(tmp_path):
    """Test write_report creates a JSON file."""
    results = [_make_result("t1", success=True)]
    report = eval_report.generate_report("suite", results, {"key": "value"})

    out_path = tmp_path / "report.json"
    eval_report.write_report(report, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["suite_name"] == "suite"
    assert data["config"] == {"key": "value"}


def test_write_report_creates_dirs(tmp_path):
    """Test write_report creates parent directories."""
    results = [_make_result("t1", success=True)]
    report = eval_report.generate_report("suite", results)

    out_path = tmp_path / "nested" / "dir" / "report.json"
    eval_report.write_report(report, out_path)

    assert out_path.exists()


def test_write_report_not_pretty(tmp_path):
    """Test write_report with pretty=False produces compact JSON."""
    results = [_make_result("t1", success=True)]
    report = eval_report.generate_report("suite", results)

    out_path = tmp_path / "report.json"
    eval_report.write_report(report, out_path, pretty=False)

    content = out_path.read_text(encoding="utf-8")
    # Compact JSON should not have indentation
    assert "\n  " not in content.rstrip("\n")


def test_load_report(tmp_path):
    """Test load_report reconstructs report from file."""
    results = [
        _make_result("t1", success=True, steps=5, tool_calls=10),
        _make_result("t2", success=False, steps=3),
    ]
    original = eval_report.generate_report("loaded_suite", results, {"model": "test"})

    path = tmp_path / "report.json"
    eval_report.write_report(original, path)
    loaded = eval_report.load_report(path)

    assert loaded.suite_name == "loaded_suite"
    assert loaded.config == {"model": "test"}
    assert len(loaded.results) == 2
    assert loaded.results[0].task_id == "t1"
    assert loaded.results[0].steps == 5
    assert loaded.metrics.total_tasks == 2


def test_load_report_with_categories(tmp_path):
    """Test load_report reconstructs by_category metrics."""
    results = [
        _make_result("t1", success=True, metadata={"category": "easy"}),
        _make_result("t2", success=False, metadata={"category": "hard"}),
    ]
    original = eval_report.generate_report("cat_suite", results)

    path = tmp_path / "report.json"
    eval_report.write_report(original, path)
    loaded = eval_report.load_report(path)

    assert "easy" in loaded.metrics.by_category
    assert "hard" in loaded.metrics.by_category
    assert isinstance(loaded.metrics.by_category["easy"], eval_metrics.EvalMetrics)


def test_roundtrip_report(tmp_path):
    """Test write then load preserves report data."""
    results = [
        _make_result(
            "t1",
            success=True,
            steps=5,
            tool_calls=10,
            duration_s=2.5,
            reflection_count=2,
            parse_errors=1,
            test_runs=3,
            metadata={"category": "test"},
        ),
    ]
    original = eval_report.generate_report(
        suite_name="roundtrip_suite",
        results=results,
        config={"sandbox": True, "max_iters": 20},
    )

    path = tmp_path / "roundtrip.json"
    eval_report.write_report(original, path)
    loaded = eval_report.load_report(path)

    assert loaded.suite_name == original.suite_name
    assert loaded.timestamp == original.timestamp
    assert loaded.config == original.config
    assert loaded.metrics.total_tasks == original.metrics.total_tasks
    assert loaded.results[0].steps == 5
    assert loaded.results[0].reflection_count == 2


def test_compare_reports_improved():
    """Test compare_reports detects improvement."""
    baseline_results = [
        _make_result("t1", success=True, steps=10, duration_s=5.0),
        _make_result("t2", success=False, steps=8, duration_s=4.0),
    ]
    current_results = [
        _make_result("t1", success=True, steps=8, duration_s=4.0),
        _make_result("t2", success=True, steps=6, duration_s=3.0),  # Now passes
    ]

    baseline = eval_report.generate_report("baseline", baseline_results)
    current = eval_report.generate_report("current", current_results)

    comparison = eval_report.compare_reports(baseline, current)

    assert comparison["baseline"]["success_rate"] == 0.5
    assert comparison["current"]["success_rate"] == 1.0
    assert comparison["delta"]["success_rate"] == 0.5
    assert comparison["delta"]["passed"] == 1
    assert comparison["improved"] is True
    assert comparison["regressed"] is False


def test_compare_reports_regressed():
    """Test compare_reports detects regression."""
    baseline_results = [
        _make_result("t1", success=True),
        _make_result("t2", success=True),
    ]
    current_results = [
        _make_result("t1", success=True),
        _make_result("t2", success=False),  # Now fails
    ]

    baseline = eval_report.generate_report("baseline", baseline_results)
    current = eval_report.generate_report("current", current_results)

    comparison = eval_report.compare_reports(baseline, current)

    assert comparison["delta"]["success_rate"] == -0.5
    assert comparison["improved"] is False
    assert comparison["regressed"] is True


def test_compare_reports_no_change():
    """Test compare_reports when no change in success rate."""
    results = [_make_result("t1", success=True)]
    baseline = eval_report.generate_report("baseline", results)
    current = eval_report.generate_report("current", results)

    comparison = eval_report.compare_reports(baseline, current)

    assert comparison["delta"]["success_rate"] == 0.0
    assert comparison["improved"] is False
    assert comparison["regressed"] is False


def test_compare_reports_deltas():
    """Test compare_reports computes all deltas correctly."""
    baseline_results = [
        _make_result("t1", success=True, steps=10, tool_calls=20, duration_s=5.0),
    ]
    current_results = [
        _make_result("t1", success=True, steps=8, tool_calls=15, duration_s=3.0),
    ]

    baseline = eval_report.generate_report("baseline", baseline_results)
    current = eval_report.generate_report("current", current_results)

    comparison = eval_report.compare_reports(baseline, current)

    assert comparison["delta"]["avg_steps"] == -2.0
    assert comparison["delta"]["avg_tool_calls"] == -5.0
    assert comparison["delta"]["avg_duration_s"] == -2.0


def test_format_comparison():
    """Test format_comparison produces readable output."""
    baseline_results = [_make_result("t1", success=True, steps=10, duration_s=5.0)]
    current_results = [_make_result("t1", success=True, steps=8, duration_s=3.0)]

    baseline = eval_report.generate_report("baseline_suite", baseline_results)
    current = eval_report.generate_report("current_suite", current_results)

    comparison = eval_report.compare_reports(baseline, current)
    output = eval_report.format_comparison(comparison)

    assert "EVALUATION COMPARISON" in output
    assert "baseline_suite" in output
    assert "current_suite" in output
    assert "Success rate:" in output
    assert "STATUS:" in output
