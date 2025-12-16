from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pprint import pp

from .llm import LLM
from .tools import RepoTools
from .prompts import system_prompt, user_prompt
from .trace import Trace


@dataclass
class AgentConfig:
  max_iters: int = 20
  max_history: int = 12
  loop_tripwire: int = 3


import json
from typing import Any, Dict, Tuple

def parse_one_json_object(raw: str) -> tuple[Dict[str, Any], str]:
  s = raw.lstrip()
  obj, idx = json.JSONDecoder().raw_decode(s)

  if not isinstance(obj, dict):
    raise ValueError(f"Model JSON must be an object. Got: {type(obj)}")

  rest = s[idx:].lstrip()
  if not rest:
    return obj, ""

  # If trailing content is another JSON value, treat as "extra actions" and ignore.
  try:
    json.JSONDecoder().raw_decode(rest)
    return obj, rest  # valid JSON remains -> model emitted multiple objects
  except json.JSONDecodeError:
    # trailing junk that's not JSON -> hard error
    raise ValueError(f"Extra trailing non-JSON content after JSON object:\n{rest[:200]}")


TOOL_NAMES = {"list_files", "read_file", "write_file", "grep"}
def coerce_action(obj: Dict[str, Any]) -> Dict[str, Any]:
  # Common mistake: {"type":"list_files","args":{...}}
  if obj.get("type") in TOOL_NAMES and "name" not in obj:
    return {"type": "tool_call", "name": obj["type"], "args": obj.get("args", {})}
  return obj


ALLOWED_KEYS = {"type", "name", "args", "summary", "changes"}
def normalize_action(obj: Dict[str, Any]) -> Dict[str, Any]:
  obj.setdefault("name", None)
  obj.setdefault("args", None)
  obj.setdefault("summary", None)
  obj.setdefault("changes", None)
  return obj


def validate_action(obj: Dict[str, Any]) -> None:
  t = obj.get("type")
  if t == "tool_call":
    if not isinstance(obj.get("name"), str) or not obj["name"]:
      raise ValueError("tool_call requires non-empty string name")
    if not isinstance(obj.get("args"), dict):
      raise ValueError("tool_call requires args object")
    if obj.get("summary") is not None or obj.get("changes") is not None:
      raise ValueError("tool_call must not include summary/changes")

  elif t == "final":
    if not isinstance(obj.get("summary"), str) or not obj["summary"]:
      raise ValueError("final requires non-empty string summary")
    if not isinstance(obj.get("changes"), list):
      raise ValueError("final requires changes array")
    for i, ch in enumerate(obj["changes"]):
      if not isinstance(ch, dict):
        raise ValueError(f"changes[{i}] must be an object")
      if set(ch.keys()) != {"path", "description"}:
        raise ValueError(f"changes[{i}] must have exactly path,description")
      if not isinstance(ch["path"], str) or not isinstance(ch["description"], str):
        raise ValueError(f"changes[{i}] path/description must be strings")
    if obj.get("name") is not None or obj.get("args") is not None:
      raise ValueError("final must not include name/args")

  else:
    raise ValueError(f"Unknown type: {t}")


def _loop_detect(history: List[Dict[str, Any]], k: int) -> bool:
  # naive: if last k tool calls have same name, you're probably stuck
  tool_calls = [h for h in history if h.get("kind") == "tool_call"]
  if len(tool_calls) < k:
    return False
  last = tool_calls[-k:]
  names = [x["name"] for x in last]
  return len(set(names)) == 1


def has_any_observation(history: List[Dict[str, Any]]) -> bool:
    return any(h.get("kind") == "observation" for h in history)


class RepoAgent:
  def __init__(self, llm: LLM, tools: RepoTools, trace: Trace, cfg: AgentConfig):
    self.llm = llm
    self.tools = tools
    self.trace = trace
    self.cfg = cfg


  def run(self, goal: str, test_cmd: List[str]) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "notes": [],
        "files_touched": [],
        "last_test": None,
    }
    history: List[Dict[str, Any]] = []

    for t in range(self.cfg.max_iters):
      compact_history = history[-self.cfg.max_history:]
      if _loop_detect(compact_history, self.cfg.loop_tripwire):
        state["notes"].append("Loop detected: change approach, inspect different evidence, then replan.")

      messages = [
          {"role": "system", "content": system_prompt()},
          {"role": "user", "content": user_prompt(goal, {"state": state, "history": compact_history})},
      ]

      self.trace.log("llm_request", {"t": t, "messages": messages})
      obj = self.llm.next_action(messages)
      self.trace.log("llm_action", {"t": t, "obj": obj})

      obj = coerce_action(obj)
      obj = normalize_action(obj)

      # HARD GATE: no final before evidence
      if obj["type"] == "final" and not has_any_observation(history):
          obj = normalize_action({"type": "tool_call", "name": "list_files",
                                  "args": {"rel_dir": ".", "max_files": 200}})
      # Now validate the final action you will execute
      validate_action(obj)

      # HARD GATE: no final before evidence
      if obj["type"] == "final" and not has_any_observation(history):
          obj = {"type": "tool_call", "name": "list_files", "args": {"rel_dir": ".", "max_files": 200}}

      typ = obj["type"]
      if typ == "final":
        self.trace.log("final", obj)
        return obj

      # From here on, it MUST be a tool_call with valid name/args
      name = obj["name"]
      args = obj["args"]
      history.append({"kind": "tool_call", "name": name, "args": args})

      if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tool_call missing/invalid name. Raw:\n{raw}")
      if not isinstance(args, dict):
        raise ValueError(f"tool_call missing/invalid args. Raw:\n{raw}")

      # dispatch action
      if name == "list_files":
        res = self.tools.list_files(**args)
      elif name == "read_file":
        res = self.tools.read_file(**args)
      elif name == "write_file":
        res = self.tools.write_file(**args)
        state["files_touched"].append(args["rel_path"])
      elif name == "grep":
        res = self.tools.grep(**args)
      elif name == "run_tests":
        # model is not allowed to call this; ignore if it tries
        res = type("X", (), {"ok": False, "output": "run_tests is driver-only", "meta": {}})()
      else:
        raise ValueError(f"Unknown tool: {name}")

      obs = {"ok": res.ok, "output": res.output[:12000], "meta": res.meta}
      history.append({"kind": "observation", "tool": name, "obs": obs})
      self.trace.log("tool_result", {"t": t, "tool": name, "args": args, "obs": obs})

      # driver runs tests occasionally (you can tune this later)
      if name in ("write_file",) and test_cmd:
        test_res = self.tools.run_tests(test_cmd)
        state["last_test"] = {"ok": test_res.ok, "output": test_res.output[:12000]}
        self.trace.log("tests", state["last_test"])
        history.append({"kind": "observation", "tool": "driver.run_tests", "obs": state["last_test"]})

    return {"type": "final", "summary": "Stopped: max iters reached.", "changes": []}
