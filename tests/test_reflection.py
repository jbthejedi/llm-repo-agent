import pytest

from llm_repo_agent.reflection import parse_reflection, ReflectionParseError
from llm_repo_agent.history import History, Observation, ObservationEvent, ReflectionEvent
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
    h.append_reflection(ReflectionEvent(notes=["note a"], next_focus=None, risks=[]), max_reflections=2, dedup_window=2)
    h.append_reflection(ReflectionEvent(notes=["note a", "note b"], next_focus=None, risks=[]), max_reflections=2, dedup_window=2)
    h.append_reflection(ReflectionEvent(notes=["note c"], next_focus="focus", risks=["risk"]), max_reflections=2, dedup_window=2)
    reflections = [e for e in h.events if isinstance(e, ReflectionEvent)]
    assert len(reflections) == 2  # capped
    all_notes = [n for e in reflections for n in e.notes]
    assert "note b" in all_notes
    assert "note c" in all_notes


def test_history_reflection_dedup_next_focus_and_risks():
    h = History()
    h.append_reflection(ReflectionEvent(notes=["note a"], next_focus="focus1", risks=["risk1"]), max_reflections=3, dedup_window=3)
    h.append_reflection(ReflectionEvent(notes=["note a", "note b"], next_focus="focus1", risks=["risk1", "risk2"]), max_reflections=3, dedup_window=3)
    reflections = [e for e in h.events if isinstance(e, ReflectionEvent)]
    assert len(reflections) == 2
    second = reflections[-1]
    assert "note b" in second.notes  # deduped to new content
    assert "risk2" in second.risks
    assert second.next_focus is None  # dedupbed out because already seen


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
    failing_obs = ObservationEvent(tool="t", observation=Observation(ok=False, output="", meta={}))
    assert rc.should_reflect(loop_triggered=False, action_observation=failing_obs, test_res=None)
    # Gating off on success when reflect_on_success False
    successful_obs = ObservationEvent(tool="t", observation=Observation(ok=True, output="", meta={}))
    assert rc.should_reflect(loop_triggered=False, action_observation=successful_obs, test_res=None) is False

    rc.run_reflection(goal="g", latest_observation={"tool": "t", "observation": {"ok": False}}, t=0)
    reflections = [e for e in history.events if isinstance(e, ReflectionEvent)]
    assert len(reflections) == 1
    assert reflections[0].notes == ["n1"]

    # Trace has reflection event
    trace_kinds = [line for line in (tmp_path / "trace.jsonl").read_text().splitlines() if '"reflection"' in line]
    assert trace_kinds
