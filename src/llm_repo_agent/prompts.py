from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .tool_schema import PROMPT_TOOL_SPEC, ALLOWED_TOOL_NAMES_TEXT


def system_prompt(tool_protocol: str = "native") -> str:
  tool_use_lines = [
    "TOOL USE:",
    "- When more evidence is needed, call exactly one tool using the tool-call mechanism.",
  ]
  if tool_protocol == "json":
    tool_use_lines.extend([
      "- Output tool calls as a JSON object in assistant content.",
      "- Tool results are provided as user messages prefixed with [tool_result].",
    ])
  else:
    tool_use_lines.append("- Do NOT output a tool_call JSON in message content.")
  tool_use_lines.append("- You may include an optional 'thought' string (<=300 chars) in tool arguments when helpful.")

  examples = [
    "EXAMPLES:",
  ]
  if tool_protocol == "json":
    examples.append(
      'Example tool call: {"type":"tool_call","name":"list_files","args":{"rel_dir":".","max_files":20}}'
    )
  else:
    examples.append("Example tool call: list_files(rel_dir='.', max_files=20, thought='Survey repo layout first.')")
  examples.append(
    'Example final: {"type":"final","summary":"Found test command: pytest","changes":[],"thought":"Goal satisfied; no edits made."}\n'
  )

  sections = [
    "You are a repo-fixing agent.",
    "You operate in a loop: choose ONE action, then wait for the tool result.",
    "",
    "\n".join(tool_use_lines),
    "",
    "FINAL OUTPUT (STRICT):",
    "- When you are done, output EXACTLY ONE JSON object as plain text.",
    "- No extra text. No markdown.",
    "- Never output multiple JSON objects.",
    "- If your response accidentally contains multiple JSON objects or trailing text, only the first object is used.",
    "",
    "FINAL JSON SHAPE:",
    '{"type":"final","summary":"...","changes":[{"path":"...","description":"..."}],"thought":"..."}',
    "",
    "\n".join(examples),
    "TOOL_CALL RULES:",
    f"- name must be EXACTLY one of: {ALLOWED_TOOL_NAMES_TEXT}",
    "- args must be an object.",
    "- NEVER call run_tests (driver-only).",
    "",
    "EVIDENCE RULE:",
    '- You MUST NOT answer with type="final" until you have called at least one tool and seen its result.',
    "- For determining test commands, you must:",
    "1) call list_files on the repo root",
    "2) if unclear, grep for: pytest, package.json, pyproject.toml, requirements, Makefile, setup.cfg, tox.ini",
    "Only then may you answer.",
    "",
    "FINAL RULES:",
    "- Use type='final' ONLY when you can answer the goal with high confidence.",
    "- 'changes' should be [] if you made no file edits.",
    "",
    "TOOLS:",
    json.dumps(PROMPT_TOOL_SPEC, indent=2),
  ]
  return "\n".join(sections).rstrip() + "\n"


def user_prompt(goal: str, state: Dict[str, Any]) -> str:
  return (
      f"GOAL:\n{goal}\n\n"
      f"STATE (compact):\n{json.dumps(state, indent=2)[:6000]}\n"
  )


@dataclass(frozen=True)
class Prompt:
  tool_protocol: str = "native"

  def system(self) -> str:
    return system_prompt(self.tool_protocol)

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
