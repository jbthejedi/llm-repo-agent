"""Evaluation harness for llm-repo-agent.

This module provides tools for running evaluation suites, collecting metrics,
and generating reports to measure agent performance across tasks.
"""

from .tasks import TaskSpec, load_suite
from .runner import EvalRunner, TaskResult
from .metrics import compute_metrics, EvalMetrics, format_metrics_summary
from .report import write_report

__all__ = [
    "TaskSpec",
    "load_suite",
    "EvalRunner",
    "TaskResult",
    "compute_metrics",
    "EvalMetrics",
    "format_metrics_summary",
    "write_report",
]
