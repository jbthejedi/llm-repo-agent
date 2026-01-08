from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
  ts: float
  kind: str
  payload: Dict[str, Any]
  run_id: str
  meta: Dict[str, Any]


@dataclass
class TracePayload:
  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class RunStartPayload(TracePayload):
  run_id: str
  goal: str


@dataclass
class RunEndPayload(TracePayload):
  run_id: str
  summary: str
  state: Dict[str, Any]


@dataclass
class LLMRequestPayload(TracePayload):
  t: int
  messages: List[Dict[str, Any]]


@dataclass
class LLMUsagePayload(TracePayload):
  t: int
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  phase: str = "action"


@dataclass
class LLMParseErrorPayload(TracePayload):
  t: int
  error: str
  raw: Any


@dataclass
class LLMActionPayload(TracePayload):
  t: int
  raw: Any
  action: Dict[str, Any]


@dataclass
class LLMTrailingTextPayload(TracePayload):
  t: int
  trailing: str


@dataclass
class DriverNotePayload(TracePayload):
  t: int
  note: str


@dataclass
class ToolResultPayload(TracePayload):
  t: int
  tool: str
  args: Dict[str, Any]
  obs: Any

  def to_dict(self) -> Dict[str, Any]:
    obs_dict = None
    if self.obs is None:
      obs_dict = None
    elif hasattr(self.obs, "to_dict"):
      obs_dict = self.obs.to_dict()
    elif is_dataclass(self.obs):
      obs_dict = asdict(self.obs)
    else:
      obs_dict = self.obs
    return {"t": self.t, "tool": self.tool, "args": self.args, "obs": obs_dict}


@dataclass
class TestsPayload(TracePayload):
  ok: bool
  output: str


@dataclass
class ReflectionRequestPayload(TracePayload):
  t: int
  messages: List[Dict[str, Any]]


@dataclass
class ReflectionPayload(TracePayload):
  t: int
  reflection: Dict[str, Any]


@dataclass
class FinalPayload(TracePayload):
  final: Dict[str, Any]


class Trace:

  def __init__(self, path: Path, run_id: str, meta: Optional[Dict[str, Any]] = None):
    self.path = path
    self.run_id = run_id
    self.meta = meta or {}
    self.path.parent.mkdir(parents=True, exist_ok=True)

  def _payload_to_dict(self, payload: Any) -> Dict[str, Any]:
    if payload is None:
      return {}
    if hasattr(payload, "to_dict"):
      return payload.to_dict()
    if is_dataclass(payload):
      return asdict(payload)
    if isinstance(payload, dict):
      return payload
    raise TypeError(f"Unsupported payload type for trace logging: {type(payload)}")

  def log(self, kind: str, payload: Any) -> None:
    payload_dict = self._payload_to_dict(payload)
    evt = TraceEvent(ts=time.time(), kind=kind, payload=payload_dict, run_id=self.run_id, meta=self.meta)
    with self.path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")

  # Helpers to iterate and reconstruct a run's events/history
  def iter_all_events(self):
    """Yield raw event dicts from the trace file in order."""
    if not self.path.exists():
      return
    with self.path.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          yield json.loads(line)
        except Exception:
          # skip malformed lines
          continue

  def iter_run_events(self, run_id: str):
    """Yield events that match a run_id in chronological order."""
    for evt in self.iter_all_events():
      if evt.get("run_id") == run_id:
        yield evt

  def get_run_history(self, run_id: str):
    """Reconstruct a history list of tool_call/observation entries for the run.

    Returns a list of dicts like the in-memory history used by the agent.
    """
    history = []
    for evt in self.iter_run_events(run_id):
      k = evt.get("kind")
      p = evt.get("payload", {})
      if k == "llm_action":
        # support older payloads (obj) and newer ones (action/raw)
        obj = p.get("obj") or p.get("action") or p.get("raw")
        if isinstance(obj, dict) and obj.get("type") == "tool_call":
          history.append({"kind": "tool_call", "name": obj.get("name"), "args": obj.get("args")})
      elif k == "tool_result":
        # payload has {"t": t, "tool": name, "args": args, "obs": obs}
        history.append({"kind": "observation", "tool": p.get("tool"), "obs": p.get("obs")})
      elif k == "tests":
        history.append({"kind": "observation", "tool": "driver.run_tests", "obs": p})
    return history
