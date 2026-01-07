from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class ToolArg:
  name: str
  type: str
  description: str
  required: bool = True


@dataclass(frozen=True)
class ToolDefinition:
  name: str
  description: str
  args: Sequence[ToolArg] = field(default_factory=list)
  driver_only: bool = False


# Single source of truth for tool definitions
TOOL_DEFINITIONS: Sequence[ToolDefinition] = (
    ToolDefinition(
        name="list_files",
        description="List files under a repo-relative directory.",
        args=(
            ToolArg(name="rel_dir", type="string", description="Directory path relative to repo root."),
            ToolArg(name="max_files", type="integer", description="Max files to list before truncating."),
        ),
    ),
    ToolDefinition(
        name="read_file",
        description="Read a text file under the repo (repo-relative path).",
        args=(
            ToolArg(name="rel_path", type="string", description="Path to the file to read."),
            ToolArg(name="max_chars", type="integer", description="Max characters to return."),
        ),
    ),
    ToolDefinition(
        name="write_file",
        description="Write/replace a file under the repo (repo-relative path).",
        args=(
            ToolArg(name="rel_path", type="string", description="Path to write (relative to repo root)."),
            ToolArg(name="content", type="string", description="Full file contents to write."),
        ),
    ),
    ToolDefinition(
        name="grep",
        description="Search for a pattern in files under a repo-relative directory.",
        args=(
            ToolArg(name="pattern", type="string", description="Literal substring to search for."),
            ToolArg(name="rel_dir", type="string", description="Directory to search under."),
            ToolArg(name="max_hits", type="integer", description="Max matches to return before truncating."),
        ),
    ),
    # Driver-only helper; included in prompt but not callable by the model.
    ToolDefinition(
        name="run_tests",
        description="Not callable by model. Driver-only.",
        args=(),
        driver_only=True,
    ),
)


TOOL_NAMES = {t.name for t in TOOL_DEFINITIONS if not t.driver_only}
TOOL_ARG_NAMES: Dict[str, List[str]] = {
    t.name: [arg.name for arg in t.args] for t in TOOL_DEFINITIONS if not t.driver_only
}
TOOL_REQUIRED_ARGS: Dict[str, List[str]] = {
    t.name: [arg.name for arg in t.args if arg.required] for t in TOOL_DEFINITIONS if not t.driver_only
}
ALLOWED_TOOL_NAMES_TEXT = ", ".join(sorted(TOOL_NAMES))


def _json_schema_type(arg_type: str) -> str:
  # passthrough today; hook for future coercion/validation if we change arg type labels
  return arg_type


def build_chat_completion_tools(defs: Sequence[ToolDefinition] = TOOL_DEFINITIONS) -> List[Dict[str, Any]]:
  """OpenAI-compatible chat-completions tool schema (function nested)."""
  tools: List[Dict[str, Any]] = []
  for tool in defs:
    if tool.driver_only:
      continue
    properties: Dict[str, Any] = {
        arg.name: {"type": _json_schema_type(arg.type), "description": arg.description} for arg in tool.args
    }
    properties["thought"] = {"type": "string"}
    required = [arg.name for arg in tool.args if arg.required]
    tools.append({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            },
        },
    })
  return tools


def _prompt_type(arg_type: str) -> str:
  return "int" if arg_type == "integer" else arg_type


def build_prompt_tool_spec(defs: Sequence[ToolDefinition] = TOOL_DEFINITIONS) -> List[Dict[str, Any]]:
  """Compact, human-readable tool list for the system prompt."""
  spec: List[Dict[str, Any]] = []
  for tool in defs:
    if tool.driver_only:
      spec.append({"name": tool.name, "args": {"note": tool.description}})
      continue
    args_map = {arg.name: _prompt_type(arg.type) for arg in tool.args}
    spec.append({"name": tool.name, "args": args_map})
  return spec


CHAT_COMPLETIONS_TOOLS = build_chat_completion_tools()
PROMPT_TOOL_SPEC = build_prompt_tool_spec()
