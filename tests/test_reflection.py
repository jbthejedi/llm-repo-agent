import pytest

from llm_repo_agent.reflection import parse_reflection, ReflectionParseError
from llm_repo_agent.history import History


def test_parse_reflection_basic():
    obj = {"notes": ["fix imports", "rerun targeted tests"], "next_focus": "check setup.cfg", "risks": ["avoid broad rewrites"]}
    ref = parse_reflection(obj)
    assert ref.notes == ["fix imports", "rerun targeted tests"]
    assert ref.next_focus == "check setup.cfg"
    assert ref.risks == ["avoid broad rewrites"]


def test_parse_reflection_requires_notes():
    with pytest.raises(ReflectionParseError):
        parse_reflection({"next_focus": "x"})


def test_history_reflection_dedup_and_cap():
    h = History()
    h.append_reflection(["note a"], None, [], max_reflections=2, dedup_window=2)
    h.append_reflection(["note a", "note b"], None, [], max_reflections=2, dedup_window=2)
    h.append_reflection(["note c"], "focus", ["risk"], max_reflections=2, dedup_window=2)
    reflections = [e for e in h.events if e.get("kind") == "reflection"]
    assert len(reflections) == 2  # capped
    all_notes = [n for e in reflections for n in e.get("notes", [])]
    assert "note b" in all_notes
    assert "note c" in all_notes
