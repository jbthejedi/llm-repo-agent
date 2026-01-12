from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from llm_repo_agent.tool_schema import TOOL_REQUIRED_ARGS, TOOL_NAMES

from .config import SFTExtractConfig


def extract_sft_samples(cfg: SFTExtractConfig) -> List[Dict[str, Any]]:
    """Extract step-level SFT samples from trace files."""
    trace_dir = cfg.trace_dir.expanduser().resolve()
    trace_files = sorted(trace_dir.rglob("*.jsonl"))
    samples: List[Dict[str, Any]] = []

    if cfg.progress:
        print(f"[sft-extract] Scanning {trace_dir}...")
        print(f"[sft-extract] Found {len(trace_files)} trace files")

    stats = {
        "traces_scanned": 0,
        "successful_runs": 0,
        "skipped_failed": 0,
        "samples": 0,
    }

    for path in trace_files:
        stats["traces_scanned"] += 1
        events = _load_events(path)
        if cfg.require_success and not _is_run_successful(events):
            stats["skipped_failed"] += 1
            if cfg.progress:
                print(f"[sft-extract] Processing {path.name}... SKIP (tests failed)")
            continue

        stats["successful_runs"] += 1
        run_samples = _extract_steps(events, cfg)
        samples.extend(run_samples)
        stats["samples"] += len(run_samples)
        if cfg.progress:
            print(f"[sft-extract] Processing {path.name}... {len(run_samples)} steps")

    if cfg.progress:
        print(f"[sft-extract] {'='*60}")
        print("[sft-extract] Summary:")
        print(f"[sft-extract]   Traces scanned: {stats['traces_scanned']}")
        print(f"[sft-extract]   Successful runs: {stats['successful_runs']}")
        print(f"[sft-extract]   Skipped (tests failed): {stats['skipped_failed']}")
        print(f"[sft-extract]   Total SFT samples: {stats['samples']}")
        print(f"[sft-extract] {'='*60}")

    return samples


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _is_run_successful(events: List[Dict[str, Any]]) -> bool:
    """Check run_end.payload.state.last_test.ok == True."""
    for evt in events:
        if evt.get("kind") != "run_end":
            continue
        payload = evt.get("payload", {})
        state = payload.get("state", {})
        last_test = state.get("last_test") or {}
        return bool(last_test.get("ok") is True)
    return False


def _extract_steps(events: List[Dict[str, Any]], cfg: SFTExtractConfig) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    last_request_messages: Optional[List[Dict[str, Any]]] = None
    pending_action: Optional[Dict[str, Any]] = None

    for evt in events:
        kind = evt.get("kind")
        payload = evt.get("payload", {})

        if kind == "llm_request":
            msgs = payload.get("messages", [])
            if isinstance(msgs, list):
                last_request_messages = msgs
            continue

        if kind == "llm_action":
            action = payload.get("action") or payload.get("obj") or payload.get("raw")
            if not isinstance(action, dict):
                continue
            if action.get("type") != "tool_call":
                continue
            if not _is_valid_tool_call(action):
                continue
            if last_request_messages is None:
                continue
            pending_action = {
                "messages": last_request_messages,
                "action": action,
            }
            continue

        if kind == "tool_result" and pending_action is not None:
            obs = payload.get("obs") or {}
            ok = obs.get("ok")
            if cfg.require_valid_tool_ok and ok is not True:
                pending_action = None
                continue

            context = _normalize_messages(pending_action["messages"], cfg.max_context_chars, cfg.output_format)
            if cfg.output_format == "native":
                tool_call_msg = _tool_call_message_from_action(pending_action["action"])
                if tool_call_msg:
                    context.append(tool_call_msg)
            else:
                tool_call_json = _serialize_tool_call(pending_action["action"])
                context.append({"role": "assistant", "content": tool_call_json})
            samples.append({"messages": context})
            pending_action = None

    return samples


def _normalize_messages(messages: List[Dict[str, Any]], max_context_chars: int, output_format: str) -> List[Dict[str, Any]]:
    if output_format == "native":
        return _normalize_messages_native(messages, max_context_chars)
    return _normalize_messages_json(messages, max_context_chars)


def _normalize_messages_native(messages: List[Dict[str, Any]], max_context_chars: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            content = _truncate(str(msg.get("content") or ""), max_context_chars)
            tool_call_id = msg.get("tool_call_id")
            entry = {"role": "tool", "content": content}
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            normalized.append(entry)
            continue
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                normalized.append({"role": "assistant", "tool_calls": tool_calls})
            continue
        if role in {"system", "user", "assistant"}:
            content = msg.get("content")
            if content is None:
                content = ""
            normalized.append({"role": role, "content": str(content)})
    return normalized


def _normalize_messages_json(messages: List[Dict[str, Any]], max_context_chars: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            content = _truncate(str(msg.get("content") or ""), max_context_chars)
            normalized.append({"role": "user", "content": f"[tool_result]\n{content}"})
            continue
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                tool_call = tool_calls[0]
                action = _tool_call_from_tool_call(tool_call)
                if action:
                    normalized.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            continue
        if role in {"system", "user", "assistant"}:
            content = msg.get("content")
            if content is None:
                content = ""
            if role == "user":
                content = _truncate(str(content), max_context_chars)
            normalized.append({"role": role, "content": str(content)})
    return normalized


def _tool_call_from_tool_call(tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    func = tool_call.get("function") or {}
    name = func.get("name")
    if not isinstance(name, str):
        return None
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"type": "tool_call", "name": name, "args": args}


def _serialize_tool_call(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False)

def _tool_call_message_from_action(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = action.get("name")
    args = action.get("args")
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    args_obj = dict(args)
    thought = action.get("thought")
    if thought:
        args_obj = dict(args_obj)
        args_obj["thought"] = thought
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args_obj, ensure_ascii=False),
                },
            }
        ],
    }


def _is_valid_tool_call(action: Dict[str, Any]) -> bool:
    name = action.get("name")
    args = action.get("args")
    if not isinstance(name, str) or name not in TOOL_NAMES:
        return False
    if not isinstance(args, dict):
        return False
    required = TOOL_REQUIRED_ARGS.get(name, [])
    for key in required:
        if key not in args:
            return False
    return True


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
