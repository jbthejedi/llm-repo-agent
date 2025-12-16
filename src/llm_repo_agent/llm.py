from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol
from openai import OpenAI
import json


class LLM(Protocol):
  def next_action(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    ...


TOOLS = [
  {
    "type": "function",
    "name": "list_files",
    "description": "List files under a repo-relative directory.",
    "parameters": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "rel_dir": {"type": "string"},
        "max_files": {"type": "integer"},
      },
      "required": ["rel_dir", "max_files"],
    },
  },
  {
    "type": "function",
    "name": "read_file",
    "description": "Read a text file under the repo (repo-relative path).",
    "parameters": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "rel_path": {"type": "string"},
        "max_chars": {"type": "integer"},
      },
      "required": ["rel_path", "max_chars"],
    },
  },
  {
    "type": "function",
    "name": "write_file",
    "description": "Write/replace a file under the repo (repo-relative path).",
    "parameters": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "rel_path": {"type": "string"},
        "content": {"type": "string"},
      },
      "required": ["rel_path", "content"],
    },
  },
  {
    "type": "function",
    "name": "grep",
    "description": "Search for a pattern in files under a repo-relative directory.",
    "parameters": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "pattern": {"type": "string"},
        "rel_dir": {"type": "string"},
        "max_hits": {"type": "integer"},
      },
      "required": ["pattern", "rel_dir", "max_hits"],
    },
  },
]


@dataclass
class OpenAIResponsesLLM:
  model: str = "gpt-4.1-mini"
  temperature: float = 0.0
  max_output_tokens: int = 600

  def __post_init__(self) -> None:
    self.client = OpenAI()

  def next_action(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = self.client.responses.create(
      model=self.model,
      input=messages,
      tools=TOOLS,
      parallel_tool_calls=False,   # prefer one call at a time :contentReference[oaicite:3]{index=3}
      max_tool_calls=1,            # hard cap :contentReference[oaicite:4]{index=4}
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      # For the “final” case, force valid JSON object in text:
      text={"format": {"type": "json_object"}},  # JSON mode :contentReference[oaicite:5]{index=5}
    )

    # 1) If the model called a tool, it will appear as a function_call item. :contentReference[oaicite:6]{index=6}
    for item in getattr(resp, "output", []) or []:
      if getattr(item, "type", None) == "function_call":
        name = item.name
        args = item.arguments
        if isinstance(args, str):
          args = json.loads(args)
        return {"type": "tool_call", "name": name, "args": args}

    # 2) Otherwise, treat output_text as the FINAL json object.
    txt = (resp.output_text or "").strip()
    if not txt:
      raise RuntimeError("Empty model response (no tool call, no text).")

    obj = json.loads(txt)
    if not isinstance(obj, dict):
      raise ValueError("Final response must be a JSON object.")
    return obj