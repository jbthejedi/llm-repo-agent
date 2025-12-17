from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from .trace import Trace


def format_ts(ts: float) -> str:
  try:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
  except Exception:
    return str(ts)


def pretty_print_events(events: Iterable[dict], max_payload_len: int = 1200, full: bool = False) -> None:
  for i, evt in enumerate(events, start=1):
    ts = evt.get("ts")
    kind = evt.get("kind")
    payload = evt.get("payload", {})
    run_id = evt.get("run_id")
    ts_s = format_ts(ts) if ts else "-"

    # If full output requested, pretty-print JSON and avoid truncation
    if full:
      payload_s = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
      payload_s = json.dumps(payload, ensure_ascii=False)
      if len(payload_s) > max_payload_len:
        payload_s = payload_s[:max_payload_len] + "..."

    # Special-case LLM requests for nicer prompt display when full is requested
    if full and kind == "llm_request":
      msgs = payload.get("messages") or []
      if msgs:
        msg_lines = []
        for m in msgs:
          role = m.get("role")
          content = m.get("content")
          # avoid dumping huge content inline; show raw content block
          msg_lines.append(f"- role: {role}\n  content:\n{content}\n")
        payload_s = "\n".join(msg_lines)

    print(f"{i:3d}. {ts_s} [{kind}] (run={run_id})\n    {payload_s}\n")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description="Inspect a run's trace and pretty-print events")
  ap.add_argument("--trace", type=str, default="runs/trace.jsonl", help="Path to trace file")
  ap.add_argument("--run", dest="run_id", required=True, help="run_id to inspect")
  ap.add_argument("--max", type=int, default=0, help="If >0, limit number of events shown")
  ap.add_argument("--full", action="store_true", help="Show full (non-truncated) payloads and pretty-print prompts")
  ap.add_argument("--kind", type=str, default=None, help="If set, only show events of this kind (e.g., llm_request)")
  ap.add_argument("--index", type=int, default=None, help="If set, show only the Nth event of the filtered kind (0-based)")
  ap.add_argument("--pretty-only-prompt", action="store_true", help="Show only the LLM prompt (no metadata). Requires --kind llm_request or infers when kind omitted and events are llm_request")
  ap.add_argument("--dump-prompt", type=str, default=None, help="Write the LLM prompt content to the given file (requires --kind llm_request and --index)")
  ap.add_argument("--prompt-with-history", action="store_true", help="Print a one-line (no '\\n') prompt together with surrounding event history. Requires --kind llm_request and --index")
  ap.add_argument("--preserve-newlines", action="store_true", help="When used with --prompt-with-history, print the prompt as an indented block preserving original newlines")
  ap.add_argument("--history-window", type=int, default=5, help="Number of events before/after the selected event to show in history")
  args = ap.parse_args(argv)

  trace_path = Path(args.trace)
  if not trace_path.exists():
    print(f"Trace file not found: {trace_path}")
    return 2

  trace = Trace(trace_path, run_id=args.run_id)

  events = list(trace.iter_run_events(args.run_id))
  if not events:
    print("No events found for run_id")
    return 0

  # Optional kind filtering
  if args.kind:
    events = [e for e in events if e.get("kind") == args.kind]
    if not events:
      print(f"No events of kind '{args.kind}' for run_id")
      return 0

  # If index specified, narrow to that single event
  if args.index is not None:
    if args.index < 0 or args.index >= len(events):
      print(f"Index out of range: {args.index}")
      return 2
    events = [events[args.index]]

  if args.max and args.max > 0:
    events = events[: args.max]

  def _compose_prompt_text(evt: dict, full: bool) -> str:
    payload = evt.get("payload", {})
    msgs = payload.get("messages") or []
    parts = []
    for m in msgs:
      role = m.get("role", "")
      content = m.get("content", "")
      if not full and len(content) > 800:
        content = content[:800] + "..."
      parts.append(f"{role}: {content}")
    return "\n\n".join(parts)

  # Handle dump-prompt: requires a single llm_request event (index)
  if args.dump_prompt:
    if args.kind != "llm_request":
      print("Error: --dump-prompt requires --kind llm_request and --index")
      return 2
    if args.index is None:
      print("Error: --dump-prompt requires --index to select a single event")
      return 2
    evt = events[0]
    prompt_text = _compose_prompt_text(evt, full=args.full)
    dump_path = Path(args.dump_prompt)
    dump_path.write_text(prompt_text)
    print(f"Wrote prompt to {dump_path}")
    return 0

  # If we only want the prompt text, print only it (no header / metadata)
  if args.pretty_only_prompt:
    only_llm = all(e.get("kind") == "llm_request" for e in events)
    if not only_llm:
      print("Error: --pretty-only-prompt only makes sense when filtering llm_request events")
      return 2
    for evt in events:
      prompt_text = _compose_prompt_text(evt, full=args.full)
      print(prompt_text)
    return 0

  # Print a one-line prompt (no newlines) together with surrounding history
  if args.prompt_with_history:
    if args.kind != "llm_request":
      print("Error: --prompt-with-history requires --kind llm_request and --index")
      return 2
    if args.index is None:
      print("Error: --prompt-with-history requires --index to select a single event")
      return 2

    # We need the full event list to get surrounding history
    all_events = list(trace.iter_run_events(args.run_id))
    # Selected event (from filtered events)
    sel_evt = events[0]
    # Find index in all_events
    sel_idx = None
    for i, ev in enumerate(all_events):
      if ev == sel_evt:
        sel_idx = i
        break
    if sel_idx is None:
      print("Error: Selected event not found in run history")
      return 2

    # Compose flattened prompt (one-line)
    def _flatten_text(s: str) -> str:
      return " ".join(s.splitlines()).replace("\t", " ").strip()

    prompt_text = _compose_prompt_text(sel_evt, full=args.full)
    flat_prompt = _flatten_text(prompt_text)

    # History window
    w = max(0, args.history_window)
    start = max(0, sel_idx - w)
    end = min(len(all_events), sel_idx + w + 1)
    history = all_events[start:end]

    # Helper to make one-line summaries for history events
    def _summarize_event(evt: dict) -> str:
      kind = evt.get("kind")
      payload = evt.get("payload", {})
      if kind == "llm_request":
        s = _compose_prompt_text(evt, full=False)
      elif kind == "tool_call":
        name = payload.get("name") or payload.get("tool") or "tool_call"
        args = payload.get("args") or payload.get("kwargs") or {}
        s = f"tool_call {name} {json.dumps(args, ensure_ascii=False)}"
      elif kind == "tool_result":
        # Pick human-friendly fields if present
        summary = payload.get("summary") or payload.get("output") or json.dumps(payload, ensure_ascii=False)
        s = f"tool_result {summary}"
      else:
        s = json.dumps(payload, ensure_ascii=False)
      # flatten to one line and truncate
      one = " ".join(s.splitlines())
      if len(one) > 300:
        one = one[:300] + "..."
      return one.replace("\t", " ")

    # Print prompt and history
    if args.preserve_newlines:
      # Print multi-line prompt block with indentation for readability
      prompt_text_block = _compose_prompt_text(sel_evt, full=args.full)
      print("PROMPT:")
      for line in prompt_text_block.splitlines():
        print(f"  {line}")
      print()
    else:
      print(f"PROMPT: {flat_prompt}\n")

    print(f"HISTORY (events {start}..{end - 1} around selected index {sel_idx}):")
    for i, he in enumerate(history, start=start):
      ts_s = format_ts(he.get("ts")) if he.get("ts") else "-"
      summary = _summarize_event(he)
      print(f" - {i}: {ts_s} [{he.get('kind')}] {summary}")

    return 0

  # Emit summary header
  print(f"Trace file: {trace_path}\nRun id: {args.run_id}\n")

  pretty_print_events(events, full=args.full)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
