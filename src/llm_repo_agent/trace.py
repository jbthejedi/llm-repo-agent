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


class Trace:

  def __init__(self, path: Path):
    self.path = path
    self.path.parent.mkdir(parents=True, exist_ok=True)

  def log(self, kind: str, payload: Dict[str, Any]) -> None:
    evt = TraceEvent(ts=time.time(), kind=kind, payload=payload)
    with self.path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")
