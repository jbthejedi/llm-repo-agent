from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .tool_schema import PROMPT_TOOL_SPEC, ALLOWED_TOOL_NAMES_TEXT


def system_prompt() -> str:
  return (
    "You are a repo-fixing agent.\n"
    "You operate in a loop: choose ONE action, then wait for the tool result.\n\n"

    "TOOL USE:\n"
    "- When more evidence is needed, call exactly one tool using the tool-call mechanism.\n"
    "- Do NOT output a tool_call JSON in message content.\n"
    "- You may include an optional 'thought' string (<=300 chars) in tool arguments when helpful.\n\n"

    "FINAL OUTPUT (STRICT):\n"
    "- When you are done, output EXACTLY ONE JSON object as plain text.\n"
    "- No extra text. No markdown.\n"
    "- Never output multiple JSON objects.\n"
    "- If your response accidentally contains multiple JSON objects or trailing text, only the first object is used.\n\n"

    "FINAL JSON SHAPE:\n"
    '{"type":"final","summary":"...","changes":[{"path":"...","description":"..."}],"thought":"..."}\n\n'

    "EXAMPLES:\n"
    "Example tool call: list_files(rel_dir='.', max_files=20, thought='Survey repo layout first.')\n"
    'Example final: {"type":"final","summary":"Found test command: pytest","changes":[],"thought":"Goal satisfied; no edits made."}\n\n'

    "TOOL_CALL RULES:\n"
    f"- name must be EXACTLY one of: {ALLOWED_TOOL_NAMES_TEXT}\n"
    "- args must be an object.\n"
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
    f"{json.dumps(PROMPT_TOOL_SPEC, indent=2)}\n"
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
