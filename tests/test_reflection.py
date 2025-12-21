import pytest

from llm_repo_agent.reflection import parse_reflection, ReflectionParseError
from llm_repo_agent.history import History
from llm_repo_agent.reflection_controller import ReflectionController, ReflectionConfig
from llm_repo_agent.trace import Trace
from llm_repo_agent.reflection import Reflection


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


def test_history_reflection_dedup_next_focus_and_risks():
    h = History()
    h.append_reflection(["note a"], "focus1", ["risk1"], max_reflections=3, dedup_window=3)
    h.append_reflection(["note a", "note b"], "focus1", ["risk1", "risk2"], max_reflections=3, dedup_window=3)
    reflections = [e for e in h.events if e.get("kind") == "reflection"]
    assert len(reflections) == 2
    second = reflections[-1]
    assert "note b" in second["notes"]  # deduped to new content
    assert "risk2" in second["risks"]
    assert second.get("next_focus") is None  # dedupbed out because already seen


def test_reflection_controller_gating_and_run(tmp_path):
    class DummyLLM:
        def __init__(self):
            self.called = False
        def reflect(self, messages):
            self.called = True
            return Reflection(notes=["n1"], next_focus="focus", risks=["r1"])

    history = History()
    trace = Trace(tmp_path / "trace.jsonl", run_id="r1")
    rc = ReflectionController(
        llm=DummyLLM(),
        trace=trace,
        history=history,
        cfg=ReflectionConfig(enable=True, max_reflections=2, reflection_dedup_window=2, reflect_on_success=False, reflection_history_window=4),
        progress_cb=lambda msg: None,
    )

    # Gating on failure
    assert rc.should_reflect(loop_triggered=False, obs={"ok": False}, test_res=None)
    # Gating off on success when reflect_on_success False
    assert rc.should_reflect(loop_triggered=False, obs={"ok": True}, test_res=None) is False

    rc.run_reflection(goal="g", latest_observation={"tool": "t", "observation": {"ok": False}}, t=0)
    reflections = [e for e in history.events if e.get("kind") == "reflection"]
    assert len(reflections) == 1
    assert reflections[0]["notes"] == ["n1"]

    # Trace has reflection event
    trace_kinds = [line for line in (tmp_path / "trace.jsonl").read_text().splitlines() if '"reflection"' in line]
    assert trace_kinds
