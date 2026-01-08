"""Preference data generation for DPO finetuning."""

from .schema import PreferencePair, PreferenceMeta, format_together_jsonl
from .score import score_rollout, RolloutScore
from .pairs import select_pair, PairResult
from .rollouts import run_rollouts, RolloutResult

__all__ = [
    "PreferencePair",
    "PreferenceMeta",
    "format_together_jsonl",
    "score_rollout",
    "RolloutScore",
    "select_pair",
    "PairResult",
    "run_rollouts",
    "RolloutResult",
]