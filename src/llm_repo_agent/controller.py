from __future__ import annotations
from typing import Any, Dict

from .tools import RepoTools, ToolResult
from .trace import Trace, DriverNotePayload, ToolResultPayload
from .history import History, Observation, ObservationEvent, DriverNoteEvent
from .tool_schema import TOOL_ARG_NAMES, TOOL_REQUIRED_ARGS


class ActionController:
    def __init__(self, tools: RepoTools, trace: Trace):
        self.tools = tools
        self.trace = trace

    def _record_result(self, name: str, args: Dict[str, Any], res: ToolResult, history: History, t: int) -> ObservationEvent:
        observation = Observation.from_tool_result(res)
        obs_event = ObservationEvent(tool=name, observation=observation)
        history.append_observation(obs_event)
        self.trace.log("tool_result", ToolResultPayload(t=t, tool=name, args=args, obs=observation))
        return obs_event

    def _invalid_args_result(self, name: str, args: Dict[str, Any], missing: list, extra: list, history: History,
                             t: int) -> ObservationEvent:
        parts = []
        if missing:
            parts.append(f"missing required args: {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected args: {', '.join(extra)}")
        msg = "Invalid tool args: " + "; ".join(parts)
        res = ToolResult(ok=False, output=msg, meta={"tool": name, "missing": missing, "extra": extra})
        return self._record_result(name, args, res, history, t)

    def execute(self, name: str, args: Dict[str, Any], history: History, t: int) -> ObservationEvent:
        allowed = TOOL_ARG_NAMES.get(name)
        required = TOOL_REQUIRED_ARGS.get(name, [])
        if allowed is not None:
            missing = [k for k in required if k not in args]
            extra = [k for k in args if k not in allowed]
            if missing or extra:
                return self._invalid_args_result(name, args, missing, extra, history, t)

        # Dispatch action to appropriate tool and return observation event
        try:
            if name == "list_files":
                res = self.tools.list_files(**args)
            elif name == "read_file":
                res = self.tools.read_file(**args)
            elif name == "write_file":
                res = self.tools.write_file(**args)
                # record file touched
                rel_path = args.get("rel_path")
                if isinstance(rel_path, str):
                    note = f"touched {rel_path}"
                    note_event = DriverNoteEvent(note=note)
                    history.append_driver_note(note_event)
                    self.trace.log("driver_note", DriverNotePayload(t=t, note=note))
            elif name == "grep":
                res = self.tools.grep(**args)
            elif name == "run_tests":
                # model is not allowed to call this; ignore if it tries
                res = ToolResult(ok=False, output="run_tests is driver-only", meta={})
            else:
                raise ValueError(f"Unknown tool: {name}")
        except TypeError as e:
            return self._record_result(
                name,
                args,
                ToolResult(ok=False, output=f"Tool call failed: {e}", meta={"tool": name, "error": str(e)}),
                history,
                t,
            )

        # run tests if write_file and test_cmd provided is responsibility of agent loop;
        return self._record_result(name, args, res, history, t)

    def execute_action(self, action, history: History, t: int) -> ObservationEvent:
        return self.execute(action.name, action.args, history, t)
