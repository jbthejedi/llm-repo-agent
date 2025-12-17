from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List


TOOL_SPEC = [
    {"name": "list_files", "args": {"rel_dir": "string", "max_files": "int"}},
    {"name": "read_file", "args": {"rel_path": "string", "max_chars": "int"}},
    {"name": "write_file", "args": {"rel_path": "string", "content": "string"}},
    {"name": "grep", "args": {"pattern": "string", "rel_dir": "string", "max_hits": "int"}},
    {"name": "run_tests", "args": {"note": "Not callable by model. Driver-only."}},
]


def system_prompt() -> str:
  return (
    "You are a repo-fixing agent.\n"
    "You operate in a loop: choose ONE action, then wait for the tool result.\n\n"

    "OUTPUT CONTRACT (STRICT):\n"
    "- Output EXACTLY ONE JSON object. No extra text. No markdown.\n"
    "- Never output multiple JSON objects.\n"
    "- If your response accidentally contains multiple JSON objects or trailing text, the agent will parse only the first JSON object and ignore the rest.\n"
    "- If more work is needed, choose the single best next tool_call and stop.\n\n"

    "ALLOWED ACTIONS:\n"
    'A) {"type":"tool_call","name":<tool_name>,"args":{...}}\n'
    'B) {"type":"final","summary":"...","changes":[{"path":"...","description":"..."}]}\n\n'

    "EXAMPLES:\n"
    'Example tool_call: {"type":"tool_call","name":"list_files","args":{"rel_dir":".","max_files":20}}\n'
    'Example final: {"type":"final","summary":"Found test command: pytest","changes":[]}\n\n'

    "TOOL_CALL RULES:\n"
    "- type must be exactly 'tool_call'.\n"
    "- name must be EXACTLY one of: list_files, read_file, write_file, grep\n"
    "- args must be an object.\n"
    "- Never put a tool name in the 'type' field.\n"
    "- NEVER call run_tests (driver-only).\n\n"

    "EVIDENCE RULE:\n"
    '- You MUST NOT answer with type="final" until you have called at least one tool and seen its result.\n'
    "- For determining test commands, you must:\n"
    "1) call list_files on the repo root\n"
    "2) if unclear, grep for: pytest, package.json, pyproject.toml, requirements, Makefile, setup.cfg, tox.ini\n"
    "Only then may you answer.\n\n"

    "FINAL RULES:\n"
    "- Use type='final' ONLY when you can answer the goal with high confidence.\n"
    "- 'changes' should be [] if you made no file edits.\n\n"

    "TOOLS:\n"
    f"{json.dumps(TOOL_SPEC, indent=2)}\n"
  )


def user_prompt(goal: str, state: Dict[str, Any]) -> str:
  return (
      f"GOAL:\n{goal}\n\n"
      f"STATE (compact):\n{json.dumps(state, indent=2)[:6000]}\n"
  )


@dataclass(frozen=True)
class Prompt:
  def system(self) -> str:
    return system_prompt()

  def user(self, goal: str, state: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return user_prompt(goal, {"state": state, "history": history})

  def compile_prompt(
      self,
      goal: str,
      state: Dict[str, Any],
      history: List[Dict[str, Any]],
  ) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": self.system(),
        },
        {
            "role": "user",
            "content": self.user(goal, state, history),
        },
    ]
