from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol, Callable
try:
  from openai import OpenAI
except Exception:
  OpenAI = None
import json
from types import SimpleNamespace

from .actions import parse_action, ActionParseError, ToolCallAction, FinalAction
from .reflection import parse_reflection, Reflection, ReflectionParseError
from .tool_schema import CHAT_COMPLETIONS_TOOLS

# Type alias for multi-turn message format
Message = Dict[str, Any]

def _init_openai_compatible_client(base_url: str | None, api_key: str | None):
  if OpenAI is None:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
      create=lambda *a, **k: SimpleNamespace(choices=[])
    )))
  try:
    if base_url:
      key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
      return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=api_key)
  except Exception:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
      create=lambda *a, **k: SimpleNamespace(choices=[])
    )))


class LLM(Protocol):

  def start_conversation(self, system_prompt: str, user_goal: str) -> None:
    ...

  def next_action(self, tool_result: str | None = None) -> Any:
    ...

  def reflect(self, messages: List[Dict[str, str]]) -> Any:
    ...


@dataclass
class ChatCompletionsLLM:
  """
  Chat Completions adaptor with proper multi-turn function calling.
  Works with OpenAI, Together, and other OpenAI-compatible providers.

  Unlike the single-turn adaptors, this class maintains conversation state
  and uses proper `assistant`/`tool` message roles for multi-turn function calling.

  Usage:
    llm = ChatCompletionsLLM(model="gpt-4.1-mini")
    llm.start_conversation(system_prompt, user_goal)
    action = llm.next_action()  # First call, no tool result
    # ... execute tool ...
    action = llm.next_action(tool_result)  # Pass result from previous tool
  """
  model: str = "gpt-4.1-mini"
  temperature: float = 0.0
  max_output_tokens: int = 600
  base_url: str | None = None  # None = OpenAI default, or Together/other URL
  api_key: str | None = None
  seed: int | None = None  # Seed for reproducibility
  tool_protocol: str = "native"

  def __post_init__(self) -> None:
    # Conversation state
    self._messages: List[Message] = []
    self._last_tool_call_id: str | None = None
    self._tool_call_counter: int = 0
    self._last_raw: Any = None
    self._last_usage: Dict[str, int] | None = None
    self._last_trailing: str | None = None
    self._last_parse_error: bool = False
    self._pending_driver_notes: List[str] = []

    # Initialize client - use OpenAI-compatible client for base_url endpoints.
    self.client = _init_openai_compatible_client(self.base_url, self.api_key)

  def start_conversation(self, system_prompt: str, user_goal: str) -> None:
    """
    Initialize conversation with system prompt and user goal.
    Must be called before next_action().
    """
    self._messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_goal},
    ]
    self._last_tool_call_id = None
    self._tool_call_counter = 0
    self._pending_driver_notes = []

  def add_driver_note(self, note: str) -> None:
    """Inject a driver note into the conversation as a system message."""
    if not note:
      return
    if not self._messages:
      return
    last = self._messages[-1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
      # Defer until the tool result is appended to keep tool_call -> tool adjacency.
      self._pending_driver_notes.append(note)
      return
    self._messages.append({
        "role": "system",
        "content": f"DRIVER NOTE:\n{note}",
    })

  def _flush_driver_notes(self) -> None:
    if not self._pending_driver_notes:
      return
    for note in self._pending_driver_notes:
      self._messages.append({
          "role": "system",
          "content": f"DRIVER NOTE:\n{note}",
      })
    self._pending_driver_notes = []

  def _generate_tool_call_id(self) -> str:
    """Generate a unique tool call ID."""
    self._tool_call_counter += 1
    return f"call_{self._tool_call_counter}"

  def _set_last_usage(self, resp: Any) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
      self._last_usage = None
      return
    if isinstance(usage, dict):
      prompt_tokens = usage.get("prompt_tokens")
      completion_tokens = usage.get("completion_tokens")
      total_tokens = usage.get("total_tokens")
    else:
      prompt_tokens = getattr(usage, "prompt_tokens", None)
      completion_tokens = getattr(usage, "completion_tokens", None)
      total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None or completion_tokens is None:
      self._last_usage = None
      return
    if total_tokens is None:
      total_tokens = prompt_tokens + completion_tokens

    self._last_usage = {
      "prompt_tokens": int(prompt_tokens),
      "completion_tokens": int(completion_tokens),
      "total_tokens": int(total_tokens),
    }

  def _decode_final(self, text: str) -> Dict[str, Any]:
    """Decode final JSON response, handling trailing text."""
    decoder = json.JSONDecoder()
    try:
      obj, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
      snippet = text[:500].replace("\n", "\\n")
      raise RuntimeError(f"Could not parse model output as JSON. Leading text: {snippet!r}")
    trailing = text[end:].strip()
    self._last_trailing = trailing if trailing else None
    if trailing:
      import warnings
      warnings.warn(
        "Model returned extra JSON objects or trailing text; using the first object and ignoring the rest.",
        UserWarning,
      )
    if not isinstance(obj, dict):
      raise ValueError("Final response must be a JSON object.")
    return obj

  def next_action(self, tool_result: str | None = None) -> ToolCallAction | FinalAction:
    """
    Get next action from the model.

    For multi-turn conversations:
    - First call: tool_result should be None
    - Subsequent calls: tool_result should be the output from the previous tool

    Args:
        tool_result: Result from the previous tool call, or None for first call

    Returns:
        ToolCallAction or FinalAction
    """
    if not self._messages:
      raise RuntimeError("Must call start_conversation() before next_action()")

    ##################################
    ########## APPEND LAST ACTION
    ##################################
    # If we have a tool result, append the tool message first
    if tool_result is not None:
      if self._last_tool_call_id is None:
        raise RuntimeError("tool_result provided but no previous tool call")
      self._messages.append({
        "role": "tool",
        "tool_call_id": self._last_tool_call_id,
        "content": tool_result,
      })
    # Append any deferred driver notes after tool results to keep proper ordering.
    self._flush_driver_notes()

    ##################################
    ########## CALL THE API
    ##################################
    api_kwargs: Dict[str, Any] = {
      "model": self.model,
      "messages": self._messages,
      "tools": CHAT_COMPLETIONS_TOOLS,
      "tool_choice": "auto",
      "temperature": self.temperature,
      "max_tokens": self.max_output_tokens,
    }
    if self.seed is not None:
      api_kwargs["seed"] = self.seed
    resp = self.client.chat.completions.create(**api_kwargs)
    self._set_last_usage(resp)

    choices = getattr(resp, "choices", []) or []
    if not choices:
      raise RuntimeError("Empty model response (no choices).")
    msg = getattr(choices[0], "message", None)
    if msg is None:
      raise RuntimeError("Model response missing message.")

    # Check for tool calls
    tool_calls = getattr(msg, "tool_calls", []) or []
    if tool_calls:
      tc = tool_calls[0]
      tc_id = getattr(tc, "id", None) or self._generate_tool_call_id()
      self._last_tool_call_id = tc_id

      func = getattr(tc, "function", None)
      name = getattr(func, "name", None)
      args_raw = getattr(func, "arguments", None)

      ##################################
      ###### APPEND ASSISTANT TOOL CALL
      ##################################
      # Append assistant message with tool_calls to maintain conversation
      self._messages.append({
        "role": "assistant",
        "tool_calls": [{
          "id": tc_id,
          "type": "function",
          "function": {
            "name": name,
            "arguments": args_raw if isinstance(args_raw, str) else json.dumps(args_raw),
          }
        }]
      })

      # Parse arguments
      args = args_raw
      if isinstance(args, str):
        args = json.loads(args)

      raw = {"type": "tool_call", "name": name, "args": args}
      self._last_raw = raw
      try:
        action = parse_action(raw)
      except ActionParseError:
        self._last_parse_error = True
        raise
      return action

    # No tool call - treat as final response
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
      content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    content = content.strip()

    if not content:
      raise RuntimeError("Empty model response (no tool call, no text).")

    # Append assistant message to maintain conversation
    self._messages.append({
      "role": "assistant",
      "content": content,
    })

    try:
      obj = self._decode_final(content)
    except Exception as e:
      self._last_raw = {"type": "invalid_json", "raw_text": content}
      self._last_parse_error = True
      raise ActionParseError(f"Could not parse model output as JSON: {e}") from e
    self._last_raw = obj
    try:
      action = parse_action(obj)
    except ActionParseError:
      self._last_parse_error = True
      raise
    return action

  def reflect(self, messages: List[Dict[str, str]]) -> Reflection:
    """
    Run reflection. For multi-turn LLM, this is a separate single-turn call
    (not part of the main conversation).
    """
    resp = self.client.chat.completions.create(
      model=self.model,
      messages=messages,
      temperature=self.temperature,
      max_tokens=self.max_output_tokens,
    )
    self._set_last_usage(resp)
    choices = getattr(resp, "choices", []) or []
    if not choices:
      raise RuntimeError("Empty reflection response.")
    msg = getattr(choices[0], "message", None)
    content = getattr(msg, "content", "") if msg else ""
    if isinstance(content, list):
      content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    content = (content or "").strip()
    if not content:
      raise RuntimeError("Empty reflection response.")

    obj = self._decode_final(content)
    self._last_reflection_raw = obj
    try:
      return parse_reflection(obj)
    except ReflectionParseError as e:
      self._last_reflection_parse_error = True
      raise


@dataclass
class JsonToolLLM:
  """
  Chat Completions adaptor that uses JSON-in-content tool calls (no native tool calling).

  The model is instructed to emit tool calls as JSON in assistant content:
    {"type":"tool_call","name":"read_file","args":{...}}
  Tool results are appended as a user message:
    "[tool_result]\\n<tool output>"
  """
  model: str = "gpt-4.1-mini"
  temperature: float = 0.0
  max_output_tokens: int = 600
  base_url: str | None = None
  api_key: str | None = None
  seed: int | None = None
  tool_protocol: str = "json"

  def __post_init__(self) -> None:
    self._messages: List[Message] = []
    self._last_raw: Any = None
    self._last_usage: Dict[str, int] | None = None
    self._last_trailing: str | None = None
    self._last_parse_error: bool = False
    self._last_reflection_raw: Any = None
    self._last_reflection_parse_error: bool = False
    self._pending_driver_notes: List[str] = []
    self.client = _init_openai_compatible_client(self.base_url, self.api_key)

  def start_conversation(self, system_prompt: str, user_goal: str) -> None:
    self._messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_goal},
    ]
    self._pending_driver_notes = []

  def add_driver_note(self, note: str) -> None:
    if not note or not self._messages:
      return
    self._messages.append({
      "role": "system",
      "content": f"DRIVER NOTE:\n{note}",
    })

  def _set_last_usage(self, resp: Any) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
      self._last_usage = None
      return
    if isinstance(usage, dict):
      prompt_tokens = usage.get("prompt_tokens")
      completion_tokens = usage.get("completion_tokens")
      total_tokens = usage.get("total_tokens")
    else:
      prompt_tokens = getattr(usage, "prompt_tokens", None)
      completion_tokens = getattr(usage, "completion_tokens", None)
      total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None or completion_tokens is None:
      self._last_usage = None
      return
    if total_tokens is None:
      total_tokens = prompt_tokens + completion_tokens

    self._last_usage = {
      "prompt_tokens": int(prompt_tokens),
      "completion_tokens": int(completion_tokens),
      "total_tokens": int(total_tokens),
    }

  def _decode_final(self, text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    try:
      obj, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
      snippet = text[:500].replace("\n", "\\n")
      raise RuntimeError(f"Could not parse model output as JSON. Leading text: {snippet!r}")
    trailing = text[end:].strip()
    self._last_trailing = trailing if trailing else None
    if trailing:
      import warnings
      warnings.warn(
        "Model returned extra JSON objects or trailing text; using the first object and ignoring the rest.",
        UserWarning,
      )
    if not isinstance(obj, dict):
      raise ValueError("Final response must be a JSON object.")
    return obj

  def next_action(self, tool_result: str | None = None) -> ToolCallAction | FinalAction:
    if not self._messages:
      raise RuntimeError("Must call start_conversation() before next_action()")

    if tool_result is not None:
      self._messages.append({
        "role": "user",
        "content": f"[tool_result]\n{tool_result}",
      })

    api_kwargs: Dict[str, Any] = {
      "model": self.model,
      "messages": self._messages,
      "temperature": self.temperature,
      "max_tokens": self.max_output_tokens,
    }
    if self.seed is not None:
      api_kwargs["seed"] = self.seed
    resp = self.client.chat.completions.create(**api_kwargs)
    self._set_last_usage(resp)

    choices = getattr(resp, "choices", []) or []
    if not choices:
      raise RuntimeError("Empty model response (no choices).")
    msg = getattr(choices[0], "message", None)
    if msg is None:
      raise RuntimeError("Model response missing message.")

    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
      content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    content = content.strip()
    if not content:
      raise RuntimeError("Empty model response (no text).")

    self._messages.append({
      "role": "assistant",
      "content": content,
    })

    try:
      obj = self._decode_final(content)
    except Exception as e:
      self._last_raw = {"type": "invalid_json", "raw_text": content}
      self._last_parse_error = True
      raise ActionParseError(f"Could not parse model output as JSON: {e}") from e
    self._last_raw = obj
    try:
      action = parse_action(obj)
    except ActionParseError:
      self._last_parse_error = True
      raise
    return action

  def reflect(self, messages: List[Dict[str, str]]) -> Reflection:
    resp = self.client.chat.completions.create(
      model=self.model,
      messages=messages,
      temperature=self.temperature,
      max_tokens=self.max_output_tokens,
    )
    self._set_last_usage(resp)
    choices = getattr(resp, "choices", []) or []
    if not choices:
      raise RuntimeError("Empty reflection response.")
    msg = getattr(choices[0], "message", None)
    content = getattr(msg, "content", "") if msg else ""
    if isinstance(content, list):
      content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    content = (content or "").strip()
    if not content:
      raise RuntimeError("Empty reflection response.")

    obj = self._decode_final(content)
    self._last_reflection_raw = obj
    try:
      return parse_reflection(obj)
    except ReflectionParseError as e:
      self._last_reflection_parse_error = True
      raise


@dataclass
class LLMConfig:
  provider: str = "openai"
  model: str | None = None
  temperature: float = 0.0
  max_output_tokens: int = 600
  together_api_key: str | None = None
  together_base_url: str | None = None
  seed: int | None = None  # Seed for reproducibility
  tool_protocol: str = "native"  # "native" | "json"


class LLMFactory:
  _registry: Dict[str, Callable[[LLMConfig], LLM]] = {}

  @classmethod
  def register(cls, provider: str, builder: Callable[[LLMConfig], LLM]) -> None:
    cls._registry[provider] = builder

  @classmethod
  def build(cls, cfg: LLMConfig) -> LLM:
    provider = cfg.provider or "openai"
    builder = cls._registry.get(provider)
    if builder is None:
      raise ValueError(f"Unknown LLM provider: {provider!r}")
    return builder(cfg)


def _build_openai(cfg: LLMConfig) -> LLM:
  model = cfg.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
  cls = JsonToolLLM if cfg.tool_protocol == "json" else ChatCompletionsLLM
  return cls(
      model=model,
      temperature=cfg.temperature,
      max_output_tokens=cfg.max_output_tokens,
      base_url=None,
      api_key=None,
      seed=cfg.seed,
      tool_protocol=cfg.tool_protocol,
  )


def _build_together(cfg: LLMConfig) -> LLM:
  model = cfg.model or os.getenv("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
  base_url = cfg.together_base_url or os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
  cls = JsonToolLLM if cfg.tool_protocol == "json" else ChatCompletionsLLM
  return cls(
      model=model,
      temperature=cfg.temperature,
      max_output_tokens=cfg.max_output_tokens,
      base_url=base_url,
      api_key=cfg.together_api_key,
      seed=cfg.seed,
      tool_protocol=cfg.tool_protocol,
  )


LLMFactory.register("openai", _build_openai)
LLMFactory.register("together", _build_together)
