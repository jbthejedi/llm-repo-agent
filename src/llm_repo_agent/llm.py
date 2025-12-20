from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol
try:
  from openai import OpenAI
except Exception:
  OpenAI = None
import json
from types import SimpleNamespace

from .actions import parse_action, ActionParseError, ToolCallAction, FinalAction
from .reflection import parse_reflection, Reflection, ReflectionParseError


class LLM(Protocol):

  def next_action(self, messages: List[Dict[str, str]]) -> Any:
    ...

  def reflect(self, messages: List[Dict[str, str]]) -> Any:
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
        "thought": {"type": "string"},
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
        "thought": {"type": "string"},
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
        "thought": {"type": "string"},
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
        "thought": {"type": "string"},
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
    try:
      self.client = OpenAI()
    except Exception:
      # Tests and some dev environments may not have OPENAI_API_KEY set. Provide a
      # minimal dummy client that yields empty responses so tests can override it.
      self.client = SimpleNamespace(responses=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(output=[], output_text="")))

  def next_action(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = self.client.responses.create(
      model=self.model,
      input=messages,
      tools=TOOLS,
      parallel_tool_calls=False,   # prefer one call at a time 
      max_tool_calls=1,            # hard cap 
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      # For the “final” case, force valid JSON object in text:
      text={"format": {"type": "json_object"}},  # JSON mode
    )

    # 1) If the model called a tool, it will appear as a function_call item.
    for item in getattr(resp, "output", []) or []:
      if getattr(item, "type", None) == "function_call":
        name = item.name
        args = item.arguments
        if isinstance(args, str):
          args = json.loads(args)
        raw = {"type": "tool_call", "name": name, "args": args}
        # record raw payload for tracing
        self._last_raw = raw
        try:
          action = parse_action(raw)
        except ActionParseError:
          self._last_parse_error = True
          raise
        return action

    # 2) Otherwise, treat output_text as the FINAL json object.
    txt = (resp.output_text or "").strip()
    if not txt:
      raise RuntimeError("Empty model response (no tool call, no text).")

    # Robustly parse only the first JSON object. If the model returns multiple
    # JSON objects or trailing text, use the first object and warn but do not
    # crash the agent.
    decoder = json.JSONDecoder()
    try:
      obj, end = decoder.raw_decode(txt)
    except json.JSONDecodeError:
      snippet = txt[:500].replace("\n", "\\n")
      raise RuntimeError(f"Could not parse model output as JSON. Leading text: {snippet!r}")

    trailing = txt[end:].strip()
    # Record trailing text on the LLM instance so the agent can trace/log it.
    self._last_trailing = trailing if trailing else None
    if trailing:
      import warnings
      warnings.warn(
        "Model returned extra JSON objects or trailing text; using the first object and ignoring the rest.",
        UserWarning,
      )

    if not isinstance(obj, dict):
      raise ValueError("Final response must be a JSON object.")

    # record raw object for tracing
    self._last_raw = obj

    # parse into typed Action; let ActionParseError bubble up so caller can handle/log
    try:
      action = parse_action(obj)
    except ActionParseError:
      self._last_parse_error = True
      raise

    return action

  def reflect(self, messages: List[Dict[str, str]]) -> Reflection:
    resp = self.client.responses.create(
      model=self.model,
      input=messages,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      text={"format": {"type": "json_object"}},  # JSON mode
    )

    txt = (getattr(resp, "output_text", "") or "").strip()
    if not txt:
      raise RuntimeError("Empty reflection response.")

    decoder = json.JSONDecoder()
    try:
      obj, end = decoder.raw_decode(txt)
    except json.JSONDecodeError:
      snippet = txt[:500].replace("\n", "\\n")
      raise RuntimeError(f"Could not parse reflection output as JSON. Leading text: {snippet!r}")

    trailing = txt[end:].strip()
    if trailing:
      import warnings
      warnings.warn("Reflection returned trailing text; using the first object.", UserWarning)

    self._last_reflection_raw = obj
    try:
      return parse_reflection(obj)
    except ReflectionParseError as e:
      self._last_reflection_parse_error = True
      raise
