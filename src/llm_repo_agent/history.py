from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class HistoryEvent:
  kind: str

  def to_dict(self) -> Dict[str, Any]:
    raise NotImplementedError


@dataclass
class ToolCallEvent(HistoryEvent):
  name: str
  args: Dict[str, Any]
  kind: str = field(init=False, default="tool_call")

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": self.kind, "name": self.name, "args": self.args}


@dataclass
class Observation:
  ok: bool
  output: str
  meta: Dict[str, Any]

  def to_dict(self) -> Dict[str, Any]:
    return {"ok": self.ok, "output": self.output, "meta": self.meta}

  @classmethod
  def from_tool_result(cls, res: Any, max_chars: int = 12000) -> "Observation":
    output = (getattr(res, "output", "") or "")[:max_chars]
    meta = getattr(res, "meta", {}) or {}
    if not isinstance(meta, dict):
      meta = {}
    return cls(ok=bool(getattr(res, "ok", False)), output=output, meta=meta)


@dataclass
class ObservationEvent(HistoryEvent):
  tool: str
  observation: Observation
  kind: str = field(init=False, default="observation")

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": self.kind, "tool": self.tool, "obs": self.observation.to_dict()}


@dataclass
class LLMActionEvent(HistoryEvent):
  obj: Dict[str, Any]
  kind: str = field(init=False, default="llm_action")

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": self.kind, "obj": self.obj}


@dataclass
class DriverNoteEvent(HistoryEvent):
  note: str
  kind: str = field(init=False, default="driver_note")

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": self.kind, "note": self.note}


@dataclass
class ReflectionEvent(HistoryEvent):
  notes: List[str]
  next_focus: Optional[str]
  risks: List[str]
  kind: str = field(init=False, default="reflection")

  def to_dict(self) -> Dict[str, Any]:
    return {
        "kind": self.kind,
        "notes": list(self.notes),
        "next_focus": self.next_focus,
        "risks": list(self.risks),
    }


class History:
  def __init__(self):
    self.events: List[HistoryEvent] = []

  def _append(self, event: HistoryEvent) -> None:
    self.events.append(event)

  def append_tool_call(self, event: ToolCallEvent) -> None:
    self._append(event)

  def append_observation(self, event: ObservationEvent) -> None:
    self._append(event)

  def append_llm_action(self, event: LLMActionEvent) -> None:
    self._append(event)

  def append_driver_note(self, event: DriverNoteEvent) -> None:
    self._append(event)

  def append_reflection(self, event: ReflectionEvent, max_reflections: Optional[int] = None, dedup_window: Optional[int] = None) -> None:
    """Add a reflection event with optional deduplication and cap."""
    if dedup_window is None and max_reflections is not None:
      dedup_window = max_reflections
    dedup_window = dedup_window or 0

    # Deduplicate against recent reflection notes
    recent_reflections = [e for e in self.events if isinstance(e, ReflectionEvent)]
    recent_slice = recent_reflections[-dedup_window:] if dedup_window > 0 else recent_reflections
    seen = set()
    for ref in recent_slice:
      for n in ref.notes or []:
        seen.add(n.strip().lower())
      nf = ref.next_focus
      if isinstance(nf, str):
        seen.add(nf.strip().lower())
      for r in ref.risks or []:
        if isinstance(r, str):
          seen.add(r.strip().lower())

    deduped_notes: List[str] = []
    for n in event.notes:
      norm = n.strip().lower()
      if norm and norm not in seen:
        deduped_notes.append(n)
        seen.add(norm)

    deduped_risks: List[str] = []
    for r in event.risks:
      norm = r.strip().lower()
      if norm and norm not in seen:
        deduped_risks.append(r)
        seen.add(norm)

    dedup_next_focus = None
    if event.next_focus:
      norm = event.next_focus.strip().lower()
      if norm and norm not in seen:
        dedup_next_focus = event.next_focus
        seen.add(norm)

    if not deduped_notes and dedup_next_focus is None and not deduped_risks:
      return

    deduped_event = ReflectionEvent(
        notes=deduped_notes or event.notes,
        next_focus=dedup_next_focus,
        risks=deduped_risks,
    )
    self._append(deduped_event)

    # Cap reflections to last max_reflections
    if max_reflections and max_reflections > 0:
      reflection_indices = [i for i, e in enumerate(self.events) if isinstance(e, ReflectionEvent)]
      overflow = len(reflection_indices) - max_reflections
      while overflow > 0 and reflection_indices:
        idx = reflection_indices.pop(0)
        self.events.pop(idx)
        reflection_indices = [i - 1 if i > idx else i for i in reflection_indices]
        overflow -= 1

  def touched_files(self) -> List[str]:
    """Return unique file paths touched by write_file observations in order."""
    seen = set()
    files: List[str] = []
    for e in self.events:
      if not isinstance(e, ObservationEvent):
        continue
      if e.tool != "write_file":
        continue
      rel = e.observation.meta.get("rel_path") or e.observation.meta.get("path")
      if isinstance(rel, str) and rel not in seen:
        seen.add(rel)
        files.append(rel)
    return files

  def last_n(self, n: int) -> List[Dict[str, Any]]:
    if n <= 0:
      return [e.to_dict() for e in self.events]
    return [e.to_dict() for e in self.events[-n:]]

  def has_any_observation(self) -> bool:
    return any(isinstance(e, ObservationEvent) for e in self.events)

  def detect_loop(self, k: int) -> bool:
    tool_calls = [e for e in self.events if isinstance(e, ToolCallEvent)]
    if len(tool_calls) < k:
      return False
    last = tool_calls[-k:]
    names = [x.name for x in last]
    return len(set(names)) == 1

  def to_prompt_list(self, max_history: int) -> List[Dict[str, Any]]:
    return self.last_n(max_history)

  def to_list(self) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in self.events]

  @classmethod
  def from_trace_events(cls, events: List[Dict[str, Any]]):
    h = cls()
    for e in events:
      k = e.get("kind")
      if k == "tool_result":
        # reconstruct observation
        payload = e.get("payload", {})
        obs = payload.get("obs") or {}
        observation = Observation(
            ok=obs.get("ok"),
            output=obs.get("output") or "",
            meta=obs.get("meta") or {},
        )
        h.append_observation(ObservationEvent(tool=payload.get("tool"), observation=observation))
      elif k == "llm_action":
        payload = e.get("payload", {})
        obj = payload.get("obj")
        if isinstance(obj, dict) and obj.get("type") == "tool_call":
          h.append_tool_call(ToolCallEvent(name=obj.get("name"), args=obj.get("args") or {}))
      elif k == "tests":
        payload = e.get("payload", {})
        observation = Observation(
            ok=payload.get("ok"),
            output=payload.get("output") or "",
            meta=payload.get("meta") if isinstance(payload, dict) else {},
        )
        h.append_observation(ObservationEvent(tool="driver.run_tests", observation=observation))
      elif k == "driver_note":
        payload = e.get("payload", {})
        note = payload.get("note") if isinstance(payload, dict) else None
        if isinstance(note, str):
          h.append_driver_note(DriverNoteEvent(note=note))
    return h
