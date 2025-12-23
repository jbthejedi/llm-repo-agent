from __future__ import annotations
from typing import Any, Dict

from .tools import RepoTools
from .trace import Trace, DriverNotePayload, ToolResultPayload
from .history import History, Observation, ObservationEvent, DriverNoteEvent


class ActionController:
    def __init__(self, tools: RepoTools, trace: Trace):
        self.tools = tools
        self.trace = trace

    def execute(self, name: str, args: Dict[str, Any], history: History, t: int) -> ObservationEvent:
        # Dispatch action to appropriate tool and return observation event
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
            res = type("X", (), {"ok": False, "output": "run_tests is driver-only", "meta": {}})()
        else:
            raise ValueError(f"Unknown tool: {name}")

        observation = Observation.from_tool_result(res)
        obs_event = ObservationEvent(tool=name, observation=observation)
        history.append_observation(obs_event)
        self.trace.log("tool_result", ToolResultPayload(t=t, tool=name, args=args, obs=observation))

        # run tests if write_file and test_cmd provided is responsibility of agent loop;
        return obs_event

    def execute_action(self, action, history: History, t: int) -> ObservationEvent:
        return self.execute(action.name, action.args, history, t)
