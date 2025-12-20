from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from .history import History


@dataclass
class TestResult:
  ok: bool
  output: str
  __test__ = False  # prevent pytest from collecting this dataclass as a test case


@dataclass
class RunSummary:
  """Compact run-level summary derived from history."""

  notes: List[str] = field(default_factory=list)
  reflection_notes: List[str] = field(default_factory=list)
  reflection_next_focus: List[str] = field(default_factory=list)
  reflection_risks: List[str] = field(default_factory=list)
  files_touched: List[str] = field(default_factory=list)
  last_test: Optional[TestResult] = None
  run_id: Optional[str] = None

  def to_dict(self) -> dict:
    return {
        "notes": list(self.notes),
        "reflection_notes": list(self.reflection_notes),
        "reflection_next_focus": list(self.reflection_next_focus),
        "reflection_risks": list(self.reflection_risks),
        "files_touched": list(self.files_touched),
        "last_test": {"ok": self.last_test.ok, "output": self.last_test.output} if self.last_test else None,
        "run_id": self.run_id,
    }


def summarize_history(history: History, run_id: Optional[str] = None) -> RunSummary:
  """Derive a compact summary from the full history."""
  notes: List[str] = []
  reflection_notes: List[str] = []
  reflection_next_focus: List[str] = []
  reflection_risks: List[str] = []
  last_test: Optional[TestResult] = None

  for e in history.events:
    kind = e.get("kind")
    if kind == "driver_note":
      note = e.get("note")
      if isinstance(note, str):
        notes.append(note)
    elif kind == "reflection":
      for n in e.get("notes") or []:
        if isinstance(n, str):
          reflection_notes.append(n)
      nf = e.get("next_focus")
      if isinstance(nf, str):
        reflection_next_focus.append(nf)
      for r in e.get("risks") or []:
        if isinstance(r, str):
          reflection_risks.append(r)
    elif kind == "observation" and e.get("tool") == "driver.run_tests":
      obs = e.get("obs") or {}
      ok = obs.get("ok")
      output = obs.get("output") or ""
      if isinstance(ok, bool):
        last_test = TestResult(ok=ok, output=output)

  files_touched = history.touched_files()

  return RunSummary(
      notes=notes,
      reflection_notes=reflection_notes,
      reflection_next_focus=reflection_next_focus,
      reflection_risks=reflection_risks,
      files_touched=files_touched,
      last_test=last_test,
      run_id=run_id,
  )
