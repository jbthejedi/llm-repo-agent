from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .tool_schema import TOOL_NAMES


@dataclass
class ToolCallAction:
  name: str
  args: Dict[str, Any]
  thought: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    d = {"type": "tool_call", "name": self.name, "args": self.args}
    if self.thought is not None:
      d["thought"] = self.thought
    return d


@dataclass
class FinalAction:
  summary: str
  changes: List[Dict[str, str]]
  thought: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    d = {"type": "final", "summary": self.summary, "changes": self.changes}
    if self.thought is not None:
      d["thought"] = self.thought
    return d


class ActionParseError(Exception):
    pass


def parse_action(obj: Any) -> Union[ToolCallAction, FinalAction]:
    """Parse a raw LLM output dict into a typed Action.

    Will coerce common mistakes (e.g., putting a tool name in `type`) and
    raise ActionParseError on invalid inputs.
    """
    if not isinstance(obj, dict):
        raise ActionParseError("Action must be a JSON object/dict")

    # Coerce common mistake: {"type":"list_files", "args":{...}}
    if obj.get("type") in TOOL_NAMES and "name" not in obj:
        obj = {
            "type": "tool_call",
            "name": obj["type"],
            "args": obj.get("args", {}),
            "thought": obj.get("thought"),
        }

    typ = obj.get("type")
    if typ == "tool_call":
        name = obj.get("name")
        args = obj.get("args")
        thought = obj.get("thought")
        if not isinstance(name, str) or not name:
            raise ActionParseError("tool_call requires non-empty string name")
        if not isinstance(args, dict):
            raise ActionParseError("tool_call requires args object")
        args_obj = dict(args)
        if thought is None and "thought" in args_obj:
            thought = args_obj.pop("thought")
        if thought is not None and not isinstance(thought, str):
            raise ActionParseError("tool_call thought must be a string if present")
        return ToolCallAction(name=name, args=args_obj, thought=thought)

    if typ == "final":
        summary = obj.get("summary")
        changes = obj.get("changes", [])
        thought = obj.get("thought")
        if not isinstance(summary, str) or not summary:
            raise ActionParseError("final requires non-empty summary string")
        if not isinstance(changes, list):
            raise ActionParseError("final requires changes array")
        for i, ch in enumerate(changes):
            if not isinstance(ch, dict) or set(ch.keys()) != {"path", "description"}:
                raise ActionParseError("each change must be {'path','description'}")
            if not isinstance(ch["path"], str) or not isinstance(ch["description"], str):
                raise ActionParseError("change path/description must be strings")
        if thought is not None and not isinstance(thought, str):
            raise ActionParseError("final thought must be a string if present")
        return FinalAction(summary=summary, changes=changes, thought=thought)

    raise ActionParseError(f"Unknown action type: {typ}")
