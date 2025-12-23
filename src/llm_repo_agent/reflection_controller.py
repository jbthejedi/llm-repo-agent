from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .history import DriverNoteEvent, ReflectionEvent, ObservationEvent
from .reflection import ReflectionParseError, compile_reflection_prompt
from .summary import summarize_history
from .trace import DriverNotePayload, ReflectionPayload, ReflectionRequestPayload


@dataclass
class ReflectionConfig:
  enable: bool = True
  max_reflections: int = 5
  reflection_dedup_window: int = 5
  reflect_on_success: bool = False
  reflection_history_window: int = 8


class ReflectionController:
  """Encapsulate reflection gating, prompting, invocation, and persistence."""

  def __init__(self, llm, trace, history, cfg: ReflectionConfig, progress_cb=None):
    self.llm = llm
    self.trace = trace
    self.history = history
    self.cfg = cfg
    self._progress = progress_cb or (lambda msg: None)

  def should_reflect(self, *, loop_triggered: bool, action_observation: Optional[Any], test_res: Optional[Any]) -> bool:
    if not self.cfg.enable:
      return False
    if loop_triggered:
      return True
    obs_ok = None
    if isinstance(action_observation, ObservationEvent):
      obs_ok = action_observation.observation.ok
    elif isinstance(action_observation, dict):
      obs_ok = action_observation.get("ok")
      if obs_ok is None and isinstance(action_observation.get("observation"), dict):
        obs_ok = action_observation["observation"].get("ok")

    if obs_ok is False:
      return True
    if test_res is not None and getattr(test_res, "ok", None) is False:
      return True
    if self.cfg.reflect_on_success and obs_ok is True:
      return True
    return False

  def run_reflection(self, *, goal: str, latest_observation: Dict[str, Any], t: int) -> None:
    if not hasattr(self.llm, "reflect"):
      note = "Reflection skipped: LLM adapter does not implement reflect()."
      note_event = DriverNoteEvent(note=note)
      self.history.append_driver_note(note_event)
      self.trace.log("driver_note", DriverNotePayload(t=t, note=note))
      self._progress(f"[reflect] iteration={t} skipped: reflect() not implemented")
      return

    ref_summary = summarize_history(self.history, run_id=self.trace.run_id).to_dict()
    recent_events = self.history.last_n(self.cfg.reflection_history_window)
    ref_messages = compile_reflection_prompt(
        goal=goal,
        summary=ref_summary,
        recent_events=recent_events,
        latest_observation=latest_observation,
    )
    self.trace.log("reflection_request", ReflectionRequestPayload(t=t, messages=ref_messages))
    self._progress(f"[reflect] iteration={t} triggered; building reflection on latest observation")
    try:
      reflection = self.llm.reflect(ref_messages)
    except ReflectionParseError as e:
      note = f"Reflection parse failed: {e}"
      note_event = DriverNoteEvent(note=note)
      self.history.append_driver_note(note_event)
      self.trace.log("driver_note", DriverNotePayload(t=t, note=note))
      self._progress(f"[reflect] iteration={t} parse failed: {e}")
      return
    except Exception as e:
      note = f"Reflection failed: {e}"
      note_event = DriverNoteEvent(note=note)
      self.history.append_driver_note(note_event)
      self.trace.log("driver_note", DriverNotePayload(t=t, note=note))
      self._progress(f"[reflect] iteration={t} failed: {e}")
      return

    reflection_event = ReflectionEvent(notes=reflection.notes, next_focus=reflection.next_focus, risks=reflection.risks)
    self.history.append_reflection(
        reflection_event,
        max_reflections=self.cfg.max_reflections,
        dedup_window=self.cfg.reflection_dedup_window,
    )
    self.trace.log("reflection", ReflectionPayload(t=t, reflection=reflection.to_dict()))
    self._progress(f"[reflect] iteration={t} notes={reflection.notes} next_focus={reflection.next_focus} risks={reflection.risks}")
