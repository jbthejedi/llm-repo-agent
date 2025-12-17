from __future__ import annotations
import argparse, os, uuid
from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.llm import OpenAIResponsesLLM
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace

from dotenv import load_dotenv

def main():
  load_dotenv()

  ap = argparse.ArgumentParser()
  ap.add_argument("--repo", type=str, required=True, help="Path to target repo")
  ap.add_argument("--goal", type=str, required=True, help="What you want the agent to do")
  ap.add_argument("--trace", type=str, default="runs/trace.jsonl")
  ap.add_argument("--test", type=str, default="", help='Test command, e.g. "python -m pytest -q"')
  args = ap.parse_args()

  repo_root = Path(args.repo).expanduser().resolve()
  tools = RepoTools(repo_root=repo_root)

  run_id = uuid.uuid4().hex[:10]
  trace = Trace(
    Path(args.trace),
    run_id=run_id,
    meta={
      "repo": str(repo_root),
      "goal": args.goal,
      "test_cmd": args.test,
      "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    },
  )

  llm = OpenAIResponsesLLM(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
  agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig())

  test_cmd = args.test.split() if args.test.strip() else []
  out = agent.run(goal=args.goal, test_cmd=test_cmd)

  # Nicely format final outputs when tests were run
  if isinstance(out, dict) and out.get("type") == "final":
    print(out.get("summary"))
    tr = out.get("test_result")
    if tr is not None:
      status = "PASSED" if tr.get("ok") else "FAILED"
      print(f"Tests: {status} - {tr.get('summary')}")
      # optional small snippet for context
      if tr.get("output_snippet"):
        snippet = tr["output_snippet"].strip().splitlines()
        print("Output snippet:", snippet[0][:200])
  else:
    print(out)


if __name__ == "__main__":
  main()
