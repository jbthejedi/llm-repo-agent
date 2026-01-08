from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .history import DriverNoteEvent, ReflectionEvent, ObservationEvent
from .reflection import ReflectionParseError, compile_reflection_prompt
from .summary import summarize_history
from .trace import DriverNotePayload, ReflectionPayload, ReflectionRequestPayload, LLMUsagePayload


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

  def should_reflect(self, *, loop_triggered: bool, obs: Optional[Any], test_res: Optional[Any]) -> bool:
    if not self.cfg.enable:
      return False
    if loop_triggered:
      return True
    obs_ok = None
    if isinstance(obs, ObservationEvent):
      obs_ok = obs.observation.ok
    elif isinstance(obs, dict):
      obs_ok = obs.get("ok")
      if obs_ok is None and isinstance(obs.get("observation"), dict):
        obs_ok = obs["observation"].get("ok")

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
    def _log_usage() -> None:
      usage = getattr(self.llm, "_last_usage", None)
      if not usage:
        return
      prompt_tokens = usage.get("prompt_tokens")
      completion_tokens = usage.get("completion_tokens")
      total_tokens = usage.get("total_tokens")
      if prompt_tokens is None or completion_tokens is None:
        return
      if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
      self.trace.log(
          "llm_usage",
          LLMUsagePayload(
              t=t,
              prompt_tokens=prompt_tokens,
              completion_tokens=completion_tokens,
              total_tokens=total_tokens,
              phase="reflection",
          ),
      )
    try:
      reflection = self.llm.reflect(ref_messages)
      _log_usage()
    except ReflectionParseError as e:
      _log_usage()
      note = f"Reflection parse failed: {e}"
      note_event = DriverNoteEvent(note=note)
      self.history.append_driver_note(note_event)
      self.trace.log("driver_note", DriverNotePayload(t=t, note=note))
      self._progress(f"[reflect] iteration={t} parse failed: {e}")
      return
    except Exception as e:
      _log_usage()
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
    self._add_reflection_note_to_message(reflection)
    self._progress(f"[reflect] iteration={t} notes={reflection.notes} next_focus={reflection.next_focus} risks={reflection.risks}")

  def _add_reflection_note_to_message(self, reflection) -> None:
    if not hasattr(self.llm, "add_driver_note"):
      return
    lines: List[str] = []
    for note in reflection.notes or []:
      lines.append(f"- {note}")
    if reflection.next_focus:
      lines.append(f"Next focus: {reflection.next_focus}")
    if reflection.risks:
      lines.append("Risks: " + "; ".join(reflection.risks))
    note_text = "\n".join(lines).strip()
    if note_text:
      self.llm.add_driver_note(note_text)
