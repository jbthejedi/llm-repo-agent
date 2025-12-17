from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pprint import pp

from .llm import LLM
from .tools import RepoTools
from .prompts import Prompt
from .trace import Trace
from .summary import RunSummary, summarize_history
from .history import History
from .controller import ActionController
from .actions import ToolCallAction, FinalAction, ActionParseError


@dataclass
class AgentConfig:
  max_iters: int = 20
  max_history: int = 12
  loop_tripwire: int = 3


# Parsing and coercion responsibilities belong to the LLM adapter.
# Adapters MUST return typed Action objects (`ToolCallAction` or `FinalAction`).
# The agent no longer attempts to coerce or validate raw dict-shaped actions.


class RepoAgent:

  def __init__(self, llm: LLM, tools: RepoTools, trace: Trace, cfg: AgentConfig):
    self.llm = llm
    self.tools = tools
    self.trace = trace
    self.cfg = cfg
    self.prompt = Prompt()

  def run(self, goal: str, test_cmd: List[str]) -> Dict[str, Any]:
    """
    Driver/Orchestrator
    """
    history = History()
    controller = ActionController(self.tools, self.trace)

    for t in range(self.cfg.max_iters):
      compact_history = history.to_prompt_list(self.cfg.max_history)
      if history.detect_loop(self.cfg.loop_tripwire):
        note = "Loop detected: change approach, inspect different evidence, then replan."
        history.append_driver_note(note)
        self.trace.log("driver_note", {"t": t, "note": note})

      summary = summarize_history(history, run_id=self.trace.run_id)
      messages = self.prompt.compile_prompt(
          goal=goal,
          state=summary.to_dict(),
          history=compact_history,
      )

      # Emit run_start at the beginning of the run (only once)
      if t == 0:
        self.trace.log("run_start", {"run_id": self.trace.run_id, "goal": goal})

      self.trace.log("llm_request", {"t": t, "messages": messages})
      # Ask LLM for next action (now returns a typed Action or raises ActionParseError)
      try:
        action = self.llm.next_action(messages)
      except ActionParseError as e:
        # Adapter failed to return a typed Action — log and surface as runtime error.
        self.trace.log("llm_parse_error", {"t": t, "error": str(e), "raw": getattr(self.llm, "_last_raw", None)})
        raise RuntimeError("LLM adapter failed to produce a valid typed Action") from e

      # Adapter MUST return a typed Action (ToolCallAction or FinalAction).
      raw = getattr(self.llm, "_last_raw", None)
      parsed_action = action
      if not isinstance(parsed_action, (ToolCallAction, FinalAction)):
        self.trace.log("llm_parse_error", {"t": t, "error": "LLM returned non-typed action", "raw": raw})
        raise RuntimeError("LLM adapter must return a typed Action (ToolCallAction or FinalAction).")

      # Log both raw and action for auditability
      self.trace.log("llm_action", {"t": t, "raw": raw, "action": parsed_action.to_dict()})
      history.append_llm_action(raw or parsed_action.to_dict())

      # If the LLM reported trailing / extra text after a JSON object, log it to the trace
      trailing = getattr(self.llm, "_last_trailing", None)
      if trailing:
        # keep the stored content reasonably sized
        self.trace.log("llm_trailing_text", {"t": t, "trailing": trailing[:12000]})
        note = "Model returned extra/trailing JSON; only the first object was used."
        history.append_driver_note(note)
        self.trace.log("driver_note", {"t": t, "note": note})

      # Now handle final
      if isinstance(parsed_action, FinalAction):

        if not history.has_any_observation(): # HARD GATE: no final before evidence
          parsed_action = ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 200})

        summary = summarize_history(history, run_id=self.trace.run_id)
        final_obj = parsed_action.to_dict()
        # If we ran tests after any write_file, include a short test result summary
        if summary.last_test is not None:
          ok = summary.last_test.ok
          out = (summary.last_test.output or "").strip()
          reason = "All tests passed." if ok else (out.splitlines()[0] if out else "Tests failed.")
          final_obj["test_result"] = {"ok": ok, "summary": reason, "output_snippet": out[:500]}

        self.trace.log("final", final_obj)
        # emit run_end
        self.trace.log("run_end", {"run_id": self.trace.run_id, "summary": final_obj.get("summary"), "state": summary.to_dict()})
        return final_obj

      # From here on, it MUST be a ToolCallAction
      history.append_tool_call(parsed_action.name, parsed_action.args)

      # dispatch via ActionController
      obs = controller.execute_action(parsed_action, history, t)

      # driver runs tests if commanded
      if parsed_action.name in ("write_file", ) and test_cmd:
        test_res = self.tools.run_tests(test_cmd)
        output_snippet = test_res.output[:12000] if test_res.output else ""
        self.trace.log("tests", {"ok": test_res.ok, "output": output_snippet})
        history.append_observation("driver.run_tests", {"ok": test_res.ok, "output": output_snippet})

    summary = {"type": "final", "summary": "Stopped: max iters reached.", "changes": []}
    # include last test summary if available
    summary_state = summarize_history(history, run_id=self.trace.run_id)
    if summary_state.last_test is not None:
      ok = summary_state.last_test.ok
      out = (summary_state.last_test.output or "").strip()
      reason = "All tests passed." if ok else (out.splitlines()[0] if out else "Tests failed.")
      summary["test_result"] = {"ok": ok, "summary": reason, "output_snippet": out[:500]}

    self.trace.log("run_end", {"run_id": self.trace.run_id, "summary": summary["summary"], "state": summary_state.to_dict()})
    return summary
