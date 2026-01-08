"""Pair selection: select best and worst rollouts to form preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .score import RolloutScore


@dataclass
class PairResult:
    """Result of pair selection.

    Attributes:
        preferred_idx: Index of the preferred (best) rollout.
        non_preferred_idx: Index of the non-preferred (worst) rollout.
        has_contrast: Whether there's meaningful contrast between best and worst.
        preferred_score: Score of the preferred rollout.
        non_preferred_score: Score of the non-preferred rollout.
    """
    preferred_idx: int
    non_preferred_idx: int
    has_contrast: bool
    preferred_score: RolloutScore
    non_preferred_score: RolloutScore


def select_pair(scores: List[RolloutScore]) -> Optional[PairResult]:
    """Select the best and worst rollouts to form a preference pair.

    Args:
        scores: List of RolloutScore objects, one per rollout.

    Returns:
        PairResult with indices of best/worst, or None if no rollouts.
    """
    if len(scores) < 2:
        return None

    # Find best (max) and worst (min) scores
    best_idx = 0
    worst_idx = 0
    best_score = scores[0]
    worst_score = scores[0]

    for i, score in enumerate(scores):
        if best_score < score:  # score > best_score
            best_score = score
            best_idx = i
        if score < worst_score:  # score < worst_score
            worst_score = score
            worst_idx = i

    # Check if there's meaningful contrast
    has_contrast = best_score != worst_score

    return PairResult(
        preferred_idx=best_idx,
        non_preferred_idx=worst_idx,
        has_contrast=has_contrast,
        preferred_score=best_score,
        non_preferred_score=worst_score,
    )