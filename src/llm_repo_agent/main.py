from __future__ import annotations
import argparse, os
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
  trace = Trace(Path(args.trace))

  llm = OpenAIResponsesLLM(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
  agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig())

  test_cmd = args.test.split() if args.test.strip() else []
  out = agent.run(goal=args.goal, test_cmd=test_cmd)
  print(out)


if __name__ == "__main__":
  main()
