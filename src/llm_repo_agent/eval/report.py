"""Report generation for evaluation results."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import TaskResult
from .metrics import EvalMetrics, compute_metrics


@dataclass
class EvalReport:
  """Complete evaluation report.

    Attributes:
        suite_name: Name of the evaluation suite.
        timestamp: ISO timestamp when the report was generated.
        metrics: Aggregate metrics.
        results: Per-task results.
        config: Configuration used for the run.
    """

  suite_name: str
  timestamp: str
  metrics: EvalMetrics
  results: List[TaskResult]
  config: Dict[str, Any]

  def to_dict(self) -> Dict[str, Any]:
    return {
        "suite_name": self.suite_name,
        "timestamp": self.timestamp,
        "metrics": self.metrics.to_dict(),
        "results": [r.to_dict() for r in self.results],
        "config": self.config,
    }


def generate_report(
    suite_name: str,
    results: List[TaskResult],
    config: Optional[Dict[str, Any]] = None,
) -> EvalReport:
  """Generate an evaluation report from results.

    Args:
        suite_name: Name of the suite that was evaluated.
        results: List of TaskResult objects.
        config: Optional configuration dict to include in the report.

    Returns:
        EvalReport with computed metrics and all results.
    """
  metrics = compute_metrics(results)
  timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

  return EvalReport(
      suite_name=suite_name,
      timestamp=timestamp,
      metrics=metrics,
      results=results,
      config=config or {},
  )


def write_report(
    report: EvalReport,
    path: Path,
    pretty: bool = True,
) -> None:
  """Write an evaluation report to a JSON file.

    Args:
        report: The EvalReport to write.
        path: Path to the output JSON file.
        pretty: Whether to format with indentation (default True).
    """
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)

  with path.open("w", encoding="utf-8") as f:
    if pretty:
      json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    else:
      json.dump(report.to_dict(), f, ensure_ascii=False)
    f.write("\n")


def load_report(path: Path) -> EvalReport:
  """Load an evaluation report from a JSON file.

    Args:
        path: Path to the report JSON file.

    Returns:
        EvalReport reconstructed from the file.
    """
  path = Path(path).expanduser().resolve()
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

  # Reconstruct TaskResult objects
  results = [TaskResult(**r) for r in data.get("results", [])]

  # Reconstruct EvalMetrics (simplified - by_category as dicts)
  metrics_data = data.get("metrics", {})
  by_category = {}
  for cat, cat_data in metrics_data.pop("by_category", {}).items():
    by_category[cat] = EvalMetrics(**cat_data)
  metrics = EvalMetrics(**metrics_data, by_category=by_category)

  return EvalReport(
      suite_name=data.get("suite_name", "unknown"),
      timestamp=data.get("timestamp", ""),
      metrics=metrics,
      results=results,
      config=data.get("config", {}),
  )


def compare_reports(
    baseline: EvalReport,
    current: EvalReport,
) -> Dict[str, Any]:
  """Compare two evaluation reports and compute deltas.

    Args:
        baseline: The baseline report (e.g., before changes).
        current: The current report (e.g., after changes).

    Returns:
        Dict with comparison metrics including deltas.
    """
  b = baseline.metrics
  c = current.metrics

  return {
      "baseline": {
          "suite": baseline.suite_name,
          "timestamp": baseline.timestamp,
          "success_rate": b.success_rate,
          "passed": b.passed,
          "total": b.total_tasks,
      },
      "current": {
          "suite": current.suite_name,
          "timestamp": current.timestamp,
          "success_rate": c.success_rate,
          "passed": c.passed,
          "total": c.total_tasks,
      },
      "delta": {
          "success_rate": c.success_rate - b.success_rate,
          "passed": c.passed - b.passed,
          "avg_steps": c.avg_steps - b.avg_steps,
          "avg_tool_calls": c.avg_tool_calls - b.avg_tool_calls,
          "avg_duration_s": c.avg_duration_s - b.avg_duration_s,
      },
      "improved": c.success_rate > b.success_rate,
      "regressed": c.success_rate < b.success_rate,
  }


def format_comparison(comparison: Dict[str, Any]) -> str:
  """Format a comparison result as a human-readable string."""
  lines = [
      "=" * 60,
      "EVALUATION COMPARISON",
      "=" * 60,
      f"Baseline: {comparison['baseline']['suite']} ({comparison['baseline']['timestamp']})",
      f"Current:  {comparison['current']['suite']} ({comparison['current']['timestamp']})",
      "-" * 40,
      f"Success rate: {comparison['baseline']['success_rate']:.1%} -> {comparison['current']['success_rate']:.1%} ({comparison['delta']['success_rate']:+.1%})",
      f"Passed:       {comparison['baseline']['passed']} -> {comparison['current']['passed']} ({comparison['delta']['passed']:+d})",
      f"Avg steps:    {comparison['delta']['avg_steps']:+.1f}",
      f"Avg duration: {comparison['delta']['avg_duration_s']:+.1f}s",
      "-" * 40,
  ]

  if comparison["improved"]:
    lines.append("STATUS: IMPROVED")
  elif comparison["regressed"]:
    lines.append("STATUS: REGRESSED")
  else:
    lines.append("STATUS: NO CHANGE")

  lines.append("=" * 60)
  return "\n".join(lines)
