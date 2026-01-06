from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pprint import pp

from .llm import LLM
from .tools import RepoTools
from .prompts import Prompt
from .trace import (
    Trace,
    DriverNotePayload,
    FinalPayload,
    LLMActionPayload,
    LLMParseErrorPayload,
    LLMRequestPayload,
    LLMTrailingTextPayload,
    RunEndPayload,
    RunStartPayload,
    TestsPayload,
)
from .summary import RunSummary, summarize_history
from .history import (
    DriverNoteEvent,
    History,
    LLMActionEvent,
    Observation,
    ObservationEvent,
    ReflectionEvent,
    ToolCallEvent,
)
from .controller import ActionController
from .actions import ToolCallAction, FinalAction, ActionParseError
from .reflection_controller import ReflectionController, ReflectionConfig

try:
  from tqdm import tqdm
except Exception:

  class _TQDMStub:

    @staticmethod
    def write(msg: str) -> None:
      print(msg)

  tqdm = _TQDMStub()


@dataclass
class AgentConfig:
  max_iters: int = 20
  max_history: int = 12
  loop_tripwire: int = 3
  progress: bool = True
  test_policy: str = "on_write"  # "on_write" | "on_final" | "never"


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
    self._progress = cfg.progress

  def _p(self, msg: str) -> None:
    if self._progress:
      tqdm.write(msg)

  def run(self, goal: str, test_cmd: List[str]) -> Dict[str, Any]:
    """
    Driver/Orchestrator
    """
    history = History()
    controller = ActionController(self.tools, self.trace)
    def _run_and_record_tests(iter_label: str):
      test_res_local = self.tools.run_tests(test_cmd)
      output_snippet_local = test_res_local.output[:12000] if test_res_local.output else ""
      self.trace.log("tests", TestsPayload(ok=test_res_local.ok, output=output_snippet_local))
      test_event_local = ObservationEvent(tool="driver.run_tests", observation=Observation.from_tool_result(test_res_local))
      history.append_observation(test_event_local)
      status_local = "PASS" if test_res_local.ok else "FAIL"
      self._p(f"[tests] iteration={iter_label} {status_local} cmd={' '.join(test_cmd)}")
      return test_res_local, test_event_local

    tests_run_for_final = False
    reflection_controller = ReflectionController(
        llm=self.llm,
        trace=self.trace,
        history=history,
        cfg=ReflectionConfig(
            enable=True,
            max_reflections=5,
            reflection_dedup_window=5,
            reflect_on_success=False,
            reflection_history_window=8,
        ),
        progress_cb=self._p,
    )

    iterator = tqdm(range(self.cfg.max_iters), disable=not self._progress, desc="agent")

    ##############################
    # DRIVER
    ##############################
    for t in iterator:
      self._p("")
      self._p(f"[iter {t}] ------------------------------")
      compact_history = history.to_prompt_list(self.cfg.max_history)
      loop_triggered = False
      if history.detect_loop(self.cfg.loop_tripwire):
        note = "Loop detected: change approach, inspect different evidence, then replan."
        note_event = DriverNoteEvent(note=note)
        history.append_driver_note(note_event)
        self.trace.log("driver_note", DriverNotePayload(t=t, note=note))
        loop_triggered = True
        self._p(f"[loop] iteration={t} tripwire triggered; adding note")

      ##############################
      # CRAFT PROMPT
      ##############################
      summary = summarize_history(history, run_id=self.trace.run_id)
      messages = self.prompt.compile_prompt(
          goal=goal,
          state=summary.to_dict(),
          history=compact_history,
      )

      # Emit run_start at the beginning of the run (only once)
      if t == 0:
        self.trace.log("run_start", RunStartPayload(run_id=self.trace.run_id, goal=goal))
        self._p(f"[start] goal='{goal}'")

      ##############################
      # SEND REQUEST TO AGENT
      ##############################
      self.trace.log("llm_request", LLMRequestPayload(t=t, messages=messages))
      # Ask LLM for next action (now returns a typed Action or raises ActionParseError)
      try:
        action = self.llm.next_action(messages)
      except ActionParseError as e:
        # Adapter failed to return a typed Action — log and surface as runtime error.
        self.trace.log("llm_parse_error", LLMParseErrorPayload(t=t, error=str(e), raw=getattr(self.llm, "_last_raw", None)),)
        raise RuntimeError("LLM adapter failed to produce a valid typed Action") from e

      ##############################
      # PARSE ACTION
      ##############################
      # Adapter MUST return a typed Action (ToolCallAction or FinalAction).
      raw = getattr(self.llm, "_last_raw", None)
      parsed_action = action
      if not isinstance(parsed_action, (ToolCallAction, FinalAction)):
        self.trace.log(
            "llm_parse_error",
            LLMParseErrorPayload(t=t, error="LLM returned non-typed action", raw=raw),
        )
        raise RuntimeError(
            "LLM adapter must return a typed Action (ToolCallAction or FinalAction).")

      # Log both raw and action for auditability
      self.trace.log("llm_action", LLMActionPayload(t=t, raw=raw, action=parsed_action.to_dict()))
      history.append_llm_action(LLMActionEvent(obj=raw or parsed_action.to_dict()))
      # progress log
      if isinstance(parsed_action, ToolCallAction):
        self._p(f"[llm] iteration={t} -> tool_call {parsed_action.name} {parsed_action.args} (thought={parsed_action.thought})")
      else:
        self._p(f"[llm] iteration={t} -> final summary='{parsed_action.summary}' (thought={parsed_action.thought})")

      # If the LLM reported trailing / extra text after a JSON object, log it to the trace
      trailing = getattr(self.llm, "_last_trailing", None)
      if trailing:
        # keep the stored content reasonably sized
        self.trace.log("llm_trailing_text", LLMTrailingTextPayload(t=t, trailing=trailing[:12000]))
        note = "Model returned extra/trailing JSON; only the first object was used."
        note_event = DriverNoteEvent(note=note)
        history.append_driver_note(note_event)
        self.trace.log("driver_note", DriverNotePayload(t=t, note=note))

      ##############################
      # HANDLE FINAL
      ##############################
      if isinstance(parsed_action, FinalAction):

        if self.cfg.test_policy == "on_final" and test_cmd and not tests_run_for_final:
          test_res, test_event = _run_and_record_tests(t)
          latest_observation = test_event.to_dict()
          tests_run_for_final = True

        if not history.has_any_observation():  # HARD GATE: no final before evidence
          parsed_action = ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 200})

        summary = summarize_history(history, run_id=self.trace.run_id)
        final_obj = parsed_action.to_dict()

        # If model didn't report changes but we touched files, supply a minimal change list
        if (not final_obj.get("changes")) and summary.files_touched:
          final_obj["changes"] = [{"path": p, "description": "Edited file"} for p in summary.files_touched]

        # If we ran tests after any write_file, include a short test result summary
        if summary.last_test is not None:
          ok = summary.last_test.ok
          out = (summary.last_test.output or "").strip()
          reason = "All tests passed." if ok else (out.splitlines()[0] if out else "Tests failed.")
          final_obj["test_result"] = {"ok": ok, "summary": reason, "output_snippet": out[:500]}

        self.trace.log("final", FinalPayload(final=final_obj))
        # emit run_end
        self.trace.log("run_end", RunEndPayload(run_id=self.trace.run_id,
                                                summary=final_obj.get("summary"),
                                                state=summary.to_dict()))
        self._p(f"[final] summary='{final_obj.get('summary')}' changes={final_obj.get('changes')}")
        return final_obj

      # From here on, it MUST be a ToolCallAction
      history.append_tool_call(ToolCallEvent(name=parsed_action.name, args=parsed_action.args))

      ##############################
      # ACTION CONTROLLER
      ##############################
      obs_event = controller.execute_action(parsed_action, history, t)
      self._p(f"[tool] iteration={t} {parsed_action.name} ok={obs_event.observation.ok} output={str(obs_event.observation.output[:160]).strip()}...")
      latest_observation = obs_event.to_dict()

      ################################
      # RUNS TESTS IF COMMANDED
      ################################
      test_res = None
      if self.cfg.test_policy == "on_write" and parsed_action.name == "write_file" and test_cmd:
        test_res, test_event = _run_and_record_tests(t)
        latest_observation = test_event.to_dict()

      ##############################
      # REFLECTION CONTROLLER
      ##############################
      if reflection_controller.should_reflect(loop_triggered=loop_triggered, obs=obs_event, test_res=test_res):
        reflection_controller.run_reflection(goal=goal, latest_observation=latest_observation, t=t)

    summary = {"type": "final", "summary": "Stopped: max iters reached.", "changes": []}
    # include last test summary if available
    if self.cfg.test_policy == "on_final" and test_cmd and not tests_run_for_final:
      _run_and_record_tests("final")
      tests_run_for_final = True

    summary_state = summarize_history(history, run_id=self.trace.run_id)
    if summary_state.last_test is not None:
      ok = summary_state.last_test.ok
      out = (summary_state.last_test.output or "").strip()
      reason = "All tests passed." if ok else (out.splitlines()[0] if out else "Tests failed.")
      summary["test_result"] = {"ok": ok, "summary": reason, "output_snippet": out[:500]}

    self.trace.log("run_end", RunEndPayload(run_id=self.trace.run_id,
                                          summary=summary["summary"],
                                          state=summary_state.to_dict()))
    return summary
