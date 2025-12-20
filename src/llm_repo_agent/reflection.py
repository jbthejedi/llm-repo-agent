from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ReflectionParseError(Exception):
  pass


@dataclass
class Reflection:
  notes: List[str] = field(default_factory=list)
  next_focus: Optional[str] = None
  risks: List[str] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "notes": list(self.notes),
        "next_focus": self.next_focus,
        "risks": list(self.risks),
    }


def parse_reflection(obj: Any) -> Reflection:
  """Parse a reflection JSON object into a Reflection dataclass."""
  if not isinstance(obj, dict):
    raise ReflectionParseError("Reflection must be a JSON object/dict")

  notes = obj.get("notes")
  next_focus = obj.get("next_focus")
  risks = obj.get("risks", [])

  if not isinstance(notes, list) or not notes:
    raise ReflectionParseError("Reflection requires notes array (1-3 items)")
  if len(notes) > 5:
    raise ReflectionParseError("Reflection notes must be short (max 5)")

  clean_notes: List[str] = []
  for n in notes:
    if not isinstance(n, str) or not n.strip():
      raise ReflectionParseError("Each note must be a non-empty string")
    clean_notes.append(n.strip())

  if next_focus is not None:
    if not isinstance(next_focus, str) or not next_focus.strip():
      raise ReflectionParseError("next_focus must be a non-empty string if provided")
    next_focus = next_focus.strip()

  clean_risks: List[str] = []
  if risks is not None:
    if not isinstance(risks, list):
      raise ReflectionParseError("risks must be a list of strings if provided")
    for r in risks:
      if not isinstance(r, str) or not r.strip():
        raise ReflectionParseError("Each risk must be a non-empty string")
      clean_risks.append(r.strip())

  return Reflection(notes=clean_notes, next_focus=next_focus, risks=clean_risks)


def reflection_system_prompt() -> str:
  return (
      "You are a reflection module.\n"
      "You DO NOT choose tools. You DO NOT edit files. You only produce durable lessons from the latest evidence.\n\n"
      "OUTPUT CONTRACT (STRICT):\n"
      "- Output EXACTLY ONE JSON object. No extra text. No markdown.\n"
      "- Fields:\n"
      '  notes: list of 1-3 short strings (<=200 chars) with actionable lessons grounded in the latest observation.\n'
      '  next_focus: optional string (<=200 chars) with the single most important next probe/fix.\n'
      '  risks: optional list of short strings for pitfalls to avoid.\n'
      "- Never emit tool calls. Never include multiple JSON objects.\n"
  )


def reflection_user_prompt(goal: str, summary: Dict[str, Any], recent_events: List[Dict[str, Any]], latest_observation: Dict[str, Any]) -> str:
  state = {
      "goal": goal,
      "summary": summary,
      "recent_events": recent_events,
      "latest_observation": latest_observation,
  }
  return json.dumps(state, ensure_ascii=False, indent=2)[:8000]


def compile_reflection_prompt(goal: str, summary: Dict[str, Any], recent_events: List[Dict[str, Any]], latest_observation: Dict[str, Any]) -> List[Dict[str, str]]:
  return [
      {"role": "system", "content": reflection_system_prompt()},
      {"role": "user", "content": reflection_user_prompt(goal, summary, recent_events, latest_observation)},
  ]
