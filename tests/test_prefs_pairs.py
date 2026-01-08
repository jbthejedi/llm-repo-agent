"""Tests for prefs/pairs.py - select_pair, PairResult."""

from llm_repo_agent.prefs.pairs import select_pair, PairResult
from llm_repo_agent.prefs.score import RolloutScore


def test_pair_result_basic():
    """Test PairResult creation and basic attributes."""
    score_a = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)
    score_b = RolloutScore(primary=0.0, steps=3, tool_calls=5, files_touched=1)

    result = PairResult(
        preferred_idx=0,
        non_preferred_idx=1,
        has_contrast=True,
        preferred_score=score_a,
        non_preferred_score=score_b,
    )

    assert result.preferred_idx == 0
    assert result.non_preferred_idx == 1
    assert result.has_contrast is True
    assert result.preferred_score == score_a
    assert result.non_preferred_score == score_b


def test_select_pair_none_for_single_score():
    """Test select_pair returns None for less than 2 scores."""
    scores = [RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)]
    result = select_pair(scores)
    assert result is None


def test_select_pair_none_for_empty():
    """Test select_pair returns None for empty list."""
    result = select_pair([])
    assert result is None


def test_select_pair_basic_contrast():
    """Test select_pair with clear pass/fail contrast."""
    scores = [
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2),  # best
        RolloutScore(primary=0.0, steps=3, tool_calls=5, files_touched=1),   # worst
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.preferred_idx == 0
    assert result.non_preferred_idx == 1
    assert result.has_contrast is True
    assert result.preferred_score.primary == 1.0
    assert result.non_preferred_score.primary == 0.0


def test_select_pair_reversed_order():
    """Test select_pair correctly finds best/worst regardless of order."""
    scores = [
        RolloutScore(primary=0.0, steps=3, tool_calls=5, files_touched=1),   # worst first
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2),  # best second
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.preferred_idx == 1  # best is at index 1
    assert result.non_preferred_idx == 0  # worst is at index 0
    assert result.has_contrast is True


def test_select_pair_no_contrast_all_pass():
    """Test select_pair with no contrast (all pass, same metrics)."""
    scores = [
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2),
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2),
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is False
    # Both scores are equal, so indices depend on implementation
    # but has_contrast should be False


def test_select_pair_no_contrast_all_fail():
    """Test select_pair with no contrast (all fail, same metrics)."""
    scores = [
        RolloutScore(primary=0.0, steps=3, tool_calls=5, files_touched=1),
        RolloutScore(primary=0.0, steps=3, tool_calls=5, files_touched=1),
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is False


def test_select_pair_contrast_via_tiebreaker():
    """Test select_pair finds contrast through tie-breakers."""
    scores = [
        RolloutScore(primary=1.0, steps=10, tool_calls=20, files_touched=5),  # worse (more steps)
        RolloutScore(primary=1.0, steps=3, tool_calls=5, files_touched=1),    # better (fewer steps)
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is True
    assert result.preferred_idx == 1  # fewer steps
    assert result.non_preferred_idx == 0  # more steps


def test_select_pair_multiple_candidates():
    """Test select_pair with multiple rollouts."""
    scores = [
        RolloutScore(primary=0.0, steps=5, tool_calls=10, files_touched=2),   # fail
        RolloutScore(primary=1.0, steps=8, tool_calls=15, files_touched=3),   # pass but slow
        RolloutScore(primary=1.0, steps=3, tool_calls=5, files_touched=1),    # pass and fast - BEST
        RolloutScore(primary=0.0, steps=2, tool_calls=3, files_touched=0),    # fail - WORST (by primary)
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is True
    assert result.preferred_idx == 2  # pass and fast
    # non_preferred is one of the failing ones (index 0 or 3)
    assert result.non_preferred_idx in [0, 3]
    assert result.preferred_score.primary == 1.0
    assert result.non_preferred_score.primary == 0.0


def test_select_pair_four_rollouts_mixed():
    """Test select_pair with realistic 4-rollout scenario."""
    scores = [
        RolloutScore(primary=1.0, steps=10, tool_calls=25, files_touched=4),  # pass, slow
        RolloutScore(primary=0.0, steps=5, tool_calls=10, files_touched=2),   # fail
        RolloutScore(primary=1.0, steps=6, tool_calls=12, files_touched=2),   # pass, fastest - BEST
        RolloutScore(primary=0.0, steps=3, tool_calls=5, files_touched=1),    # fail
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is True
    assert result.preferred_idx == 2  # best passing run
    assert result.preferred_score.primary == 1.0
    assert result.non_preferred_score.primary == 0.0


def test_select_pair_tiebreaker_ordering():
    """Test that tie-breakers are applied in correct order: steps > tool_calls > files."""
    # All pass, but differ in tie-breaker metrics
    scores = [
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=3),  # same steps, more calls
        RolloutScore(primary=1.0, steps=5, tool_calls=8, files_touched=3),   # same steps, fewer calls - BEST
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is True
    assert result.preferred_idx == 1  # fewer tool calls wins


def test_select_pair_files_touched_tiebreaker():
    """Test files_touched as final tie-breaker."""
    scores = [
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=5),  # more files - WORST
        RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2),  # fewer files - BEST
    ]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is True
    assert result.preferred_idx == 1  # fewer files
    assert result.non_preferred_idx == 0


def test_select_pair_two_equal_scores():
    """Test select_pair with exactly two equal scores."""
    score = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)
    scores = [score, score]
    result = select_pair(scores)

    assert result is not None
    assert result.has_contrast is False
    assert result.preferred_idx == 0  # first one picked as best
    assert result.non_preferred_idx == 0  # same as best (no difference)