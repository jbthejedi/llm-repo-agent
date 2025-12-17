from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import itertools


@dataclass
class ToolCallEvent:
  name: str
  args: Dict[str, Any]

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": "tool_call", "name": self.name, "args": self.args}


@dataclass
class ObservationEvent:
  tool: str
  obs: Dict[str, Any]

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": "observation", "tool": self.tool, "obs": self.obs}


@dataclass
class LLMActionEvent:
  obj: Dict[str, Any]

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": "llm_action", "obj": self.obj}


@dataclass
class DriverNoteEvent:
  note: str

  def to_dict(self) -> Dict[str, Any]:
    return {"kind": "driver_note", "note": self.note}


class History:
  def __init__(self):
    self.events: List[Dict[str, Any]] = []

  def append_tool_call(self, name: str, args: Dict[str, Any]) -> None:
    self.events.append(ToolCallEvent(name=name, args=args).to_dict())

  def append_observation(self, tool: str, obs: Dict[str, Any]) -> None:
    self.events.append(ObservationEvent(tool=tool, obs=obs).to_dict())

  def append_llm_action(self, obj: Dict[str, Any]) -> None:
    self.events.append(LLMActionEvent(obj=obj).to_dict())

  def append_driver_note(self, note: str) -> None:
    self.events.append(DriverNoteEvent(note=note).to_dict())

  def touched_files(self) -> List[str]:
    """Return unique file paths touched by write_file observations in order."""
    seen = set()
    files: List[str] = []
    for e in self.events:
      if e.get("kind") != "observation":
        continue
      if e.get("tool") != "write_file":
        continue
      obs = e.get("obs") or {}
      rel = None
      meta = obs.get("meta") if isinstance(obs, dict) else None
      if isinstance(meta, dict):
        rel = meta.get("rel_path") or meta.get("path")
      if isinstance(rel, str) and rel not in seen:
        seen.add(rel)
        files.append(rel)
    return files

  def last_n(self, n: int) -> List[Dict[str, Any]]:
    if n <= 0:
      return list(self.events)
    return self.events[-n:]

  def has_any_observation(self) -> bool:
    return any(e.get("kind") == "observation" for e in self.events)

  def detect_loop(self, k: int) -> bool:
    tool_calls = [e for e in self.events if e.get("kind") == "tool_call"]
    if len(tool_calls) < k:
      return False
    last = tool_calls[-k:]
    names = [x["name"] for x in last]
    return len(set(names)) == 1

  def to_prompt_list(self, max_history: int) -> List[Dict[str, Any]]:
    return self.last_n(max_history)

  def to_list(self) -> List[Dict[str, Any]]:
    return list(self.events)

  @classmethod
  def from_trace_events(cls, events: List[Dict[str, Any]]):
    h = cls()
    for e in events:
      k = e.get("kind")
      if k == "tool_result":
        # reconstruct observation
        payload = e.get("payload", {})
        h.append_observation(payload.get("tool"), payload.get("obs"))
      elif k == "llm_action":
        payload = e.get("payload", {})
        obj = payload.get("obj")
        if isinstance(obj, dict) and obj.get("type") == "tool_call":
          h.append_tool_call(obj.get("name"), obj.get("args"))
      elif k == "tests":
        payload = e.get("payload", {})
        h.append_observation("driver.run_tests", payload)
      elif k == "driver_note":
        payload = e.get("payload", {})
        note = payload.get("note") if isinstance(payload, dict) else None
        if isinstance(note, str):
          h.append_driver_note(note)
    return h
