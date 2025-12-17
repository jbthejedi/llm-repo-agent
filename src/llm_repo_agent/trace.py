from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
  ts: float
  kind: str
  payload: Dict[str, Any]
  run_id: str
  meta: Dict[str, Any]


class Trace:

  def __init__(self, path: Path, run_id: str, meta: Optional[Dict[str, Any]] = None):
    self.path = path
    self.run_id = run_id
    self.meta = meta or {}
    self.path.parent.mkdir(parents=True, exist_ok=True)

  def log(self, kind: str, payload: Dict[str, Any]) -> None:
    evt = TraceEvent(ts=time.time(), kind=kind, payload=payload, run_id=self.run_id, meta=self.meta)
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
