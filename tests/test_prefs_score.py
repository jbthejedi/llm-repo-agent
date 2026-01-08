"""Tests for prefs/score.py - RolloutScore, score_rollout."""

from llm_repo_agent.prefs.score import RolloutScore, score_rollout
from llm_repo_agent.eval.runner import TaskResult


def test_rollout_score_basic():
    """Test RolloutScore creation and basic attributes."""
    score = RolloutScore(
        primary=1.0,
        steps=5,
        tool_calls=10,
        files_touched=2,
    )
    assert score.primary == 1.0
    assert score.steps == 5
    assert score.tool_calls == 10
    assert score.files_touched == 2


def test_rollout_score_comparison_primary_wins():
    """Test that higher primary score wins regardless of tie-breakers."""
    passing = RolloutScore(primary=1.0, steps=20, tool_calls=50, files_touched=10)
    failing = RolloutScore(primary=0.0, steps=1, tool_calls=1, files_touched=1)

    # passing > failing even though failing has better tie-breakers
    assert failing < passing
    assert not (passing < failing)


def test_rollout_score_comparison_tiebreaker_steps():
    """Test that fewer steps wins when primary is tied."""
    fewer_steps = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)
    more_steps = RolloutScore(primary=1.0, steps=10, tool_calls=10, files_touched=2)

    assert more_steps < fewer_steps
    assert not (fewer_steps < more_steps)


def test_rollout_score_comparison_tiebreaker_tool_calls():
    """Test that fewer tool_calls wins when primary and steps are tied."""
    fewer_calls = RolloutScore(primary=1.0, steps=5, tool_calls=8, files_touched=2)
    more_calls = RolloutScore(primary=1.0, steps=5, tool_calls=12, files_touched=2)

    assert more_calls < fewer_calls
    assert not (fewer_calls < more_calls)


def test_rollout_score_comparison_tiebreaker_files_touched():
    """Test that fewer files_touched wins when other metrics are tied."""
    fewer_files = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=1)
    more_files = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=3)

    assert more_files < fewer_files
    assert not (fewer_files < more_files)


def test_rollout_score_equality():
    """Test RolloutScore equality comparison."""
    score1 = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)
    score2 = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)
    score3 = RolloutScore(primary=1.0, steps=6, tool_calls=10, files_touched=2)

    assert score1 == score2
    assert not (score1 == score3)
    assert score1 != score3


def test_rollout_score_equality_not_implemented():
    """Test RolloutScore equality with non-RolloutScore returns NotImplemented."""
    score = RolloutScore(primary=1.0, steps=5, tool_calls=10, files_touched=2)
    result = score.__eq__("not a score")
    assert result is NotImplemented


def test_score_rollout_passing_tests():
    """Test score_rollout with passing tests."""
    result = TaskResult(
        task_id="test_task",
        run_id="run123",
        success=True,
        steps=5,
        tool_calls=10,
        files_touched=["file1.py", "file2.py"],
    )
    score = score_rollout(result)

    assert score.primary == 1.0
    assert score.steps == 5
    assert score.tool_calls == 10
    assert score.files_touched == 2


def test_score_rollout_failing_tests():
    """Test score_rollout with failing tests."""
    result = TaskResult(
        task_id="test_task",
        run_id="run123",
        success=False,
        steps=3,
        tool_calls=5,
        files_touched=["file1.py"],
    )
    score = score_rollout(result)

    assert score.primary == 0.0
    assert score.steps == 3
    assert score.tool_calls == 5
    assert score.files_touched == 1


def test_score_rollout_no_test_result():
    """Test score_rollout when success is None (no tests run)."""
    result = TaskResult(
        task_id="test_task",
        run_id="run123",
        success=None,  # No test result
        steps=2,
        tool_calls=3,
        files_touched=[],
    )
    score = score_rollout(result)

    # None is treated as failure
    assert score.primary == 0.0
    assert score.steps == 2
    assert score.tool_calls == 3
    assert score.files_touched == 0


def test_score_rollout_empty_files():
    """Test score_rollout with no files touched."""
    result = TaskResult(
        task_id="test_task",
        run_id="run123",
        success=True,
        steps=1,
        tool_calls=2,
        files_touched=[],
    )
    score = score_rollout(result)

    assert score.files_touched == 0


def test_score_rollout_many_files():
    """Test score_rollout with many files touched."""
    result = TaskResult(
        task_id="test_task",
        run_id="run123",
        success=True,
        steps=10,
        tool_calls=20,
        files_touched=["a.py", "b.py", "c.py", "d.py", "e.py"],
    )
    score = score_rollout(result)

    assert score.files_touched == 5


def test_score_rollout_comparison_integration():
    """Test that scores from score_rollout compare correctly."""
    passing_quick = TaskResult(
        task_id="t1", run_id="r1", success=True, steps=3, tool_calls=5, files_touched=["a.py"]
    )
    passing_slow = TaskResult(
        task_id="t2", run_id="r2", success=True, steps=10, tool_calls=20, files_touched=["a.py", "b.py"]
    )
    failing = TaskResult(
        task_id="t3", run_id="r3", success=False, steps=1, tool_calls=1, files_touched=[]
    )

    score_quick = score_rollout(passing_quick)
    score_slow = score_rollout(passing_slow)
    score_fail = score_rollout(failing)

    # Both passing beat failing
    assert score_fail < score_quick
    assert score_fail < score_slow

    # Quick beats slow (fewer steps)
    assert score_slow < score_quick