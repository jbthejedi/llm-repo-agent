from pathlib import Path

from llm_repo_agent.controller import ActionController
from llm_repo_agent.history import History
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace


def test_controller_handles_missing_args(tmp_path: Path):
    trace = Trace(tmp_path / "trace.jsonl", run_id="run1")
    tools = RepoTools(repo_root=tmp_path)
    controller = ActionController(tools=tools, trace=trace)
    history = History()

    obs = controller.execute("grep", {}, history, t=1)
    assert obs.observation.ok is False
    assert "missing required args" in obs.observation.output
