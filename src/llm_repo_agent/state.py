from __future__ import annotations
"""Backwards-compatible re-export for RunSummary utilities."""

from .summary import RunSummary, TestResult, summarize_history

__all__ = ["RunSummary", "TestResult", "summarize_history"]
