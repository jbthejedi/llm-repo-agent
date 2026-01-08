from __future__ import annotations
import argparse
import os
import uuid
from pathlib import Path
from typing import Optional

import llm_repo_agent.agent as agent_module
import llm_repo_agent.llm as llm_module
import llm_repo_agent.tools as tools_module
import llm_repo_agent.trace as trace_module
import llm_repo_agent.sandbox as sandbox_module
import llm_repo_agent.eval.tasks as eval_tasks
import llm_repo_agent.eval.runner as eval_runner
import llm_repo_agent.eval.metrics as eval_metrics
import llm_repo_agent.eval.report as eval_report
import llm_repo_agent.prefs.rollouts as prefs_rollouts
from dotenv import load_dotenv


def cmd_run(args):
  """Run the agent on a single repo with a goal."""
  repo_root = Path(args.repo).expanduser().resolve()
  sandbox: Optional[sandbox_module.Sandbox] = None
  tools_root = repo_root
  if args.sandbox:
    sandbox_dest = Path(args.sandbox_dir).expanduser() if args.sandbox_dir else None
    sandbox = sandbox_module.materialize_repo_sandbox(repo_root, sandbox_dest)
    tools_root = sandbox.root
    print(f"[sandbox] using workspace at {tools_root}")

  tools = tools_module.RepoTools(repo_root=tools_root)

  run_id = uuid.uuid4().hex[:10]
  llm_cfg = llm_module.LLMConfig(
      provider=args.llm_provider,
      model=args.model,
      together_api_key=args.together_api_key,
  )
  llm = llm_module.LLMFactory.build(llm_cfg)
  model_name = getattr(llm, "model", None) or (args.model or "unknown")
  print(f"[llm] provider={args.llm_provider} model={model_name}")

  trace = trace_module.Trace(
      Path(args.trace),
      run_id=run_id,
      meta={
          "repo": str(repo_root),
          "goal": args.goal,
          "test_cmd": args.test,
          "model": model_name,
          "provider": args.llm_provider,
      },
  )
  agent = agent_module.RepoAgent(llm=llm, tools=tools, trace=trace, cfg=agent_module.AgentConfig(test_policy=args.test_policy))

  test_cmd = args.test.split() if args.test.strip() else []
  out = None
  try:
    out = agent.run(goal=args.goal, test_cmd=test_cmd)
  finally:
    if sandbox and not args.keep_sandbox:
      sandbox_module.cleanup_sandbox(sandbox)
      print(f"[sandbox] cleaned up {sandbox.root}")

  _print_final_output(out)


def cmd_eval(args):
  """Run an evaluation suite."""
  suite = eval_tasks.load_suite(Path(args.suite))
  print(f"[eval] Loaded suite: {suite.name} ({len(suite.tasks)} tasks)")
  if suite.description:
    print(f"[eval] Description: {suite.description}")

  cfg = eval_runner.EvalConfig(
      trace_dir=Path(args.trace_dir),
      sandbox=args.sandbox,
      keep_sandbox=args.keep_sandbox,
      test_policy=args.test_policy,
      max_iters=args.max_iters,
      model=args.model,
      llm_provider=args.llm_provider,
      together_api_key=args.together_api_key,
      progress=not args.quiet,
  )

  runner = eval_runner.EvalRunner(cfg=cfg)
  results = runner.run_suite(suite)

  # Compute and display metrics
  metrics = eval_metrics.compute_metrics(results)
  print("\n" + eval_metrics.format_metrics_summary(metrics))

  # Write report
  if args.report:
    report = eval_report.generate_report(
        suite_name=suite.name,
        results=results,
        config={
            "test_policy": args.test_policy,
            "max_iters": args.max_iters,
            "model": args.model or "auto",
            "provider": args.llm_provider,
            "sandbox": args.sandbox,
        },
    )
    eval_report.write_report(report, Path(args.report))
    print(f"\n[eval] Report written to: {args.report}")


def cmd_prefs(args):
  """Generate preference data for DPO finetuning."""
  cfg = prefs_rollouts.PrefsConfig(
      trace_dir=Path(args.trace_dir),
      out_path=Path(args.out),
      write_mode=args.data_write_mode,
      rollouts=args.rollouts,
      sandbox=args.sandbox,
      keep_sandbox=args.keep_sandbox,
      test_policy=args.test_policy,
      max_iters=args.max_iters,
      model=args.model,
      llm_provider=args.llm_provider,
      together_api_key=args.together_api_key,
      temperature=args.temperature,
      base_seed=args.seed,
      progress=not args.quiet,
  )

  prefs_rollouts.run_rollouts(Path(args.suite), cfg)


def _print_final_output(out):
  """Nicely format final outputs when tests were run."""
  if isinstance(out, dict) and out.get("type") == "final":
    print(out.get("summary"))
    changes = out.get("changes") or []
    if changes:
      print("Changes:")
      for ch in changes:
        path = ch.get("path")
        desc = ch.get("description")
        if path and desc:
          print(f"- {path}: {desc}")
        elif path:
          print(f"- {path}")
    tr = out.get("test_result")
    if tr is not None:
      status = "PASSED" if tr.get("ok") else "FAILED"
      print(f"Tests: {status} - {tr.get('summary')}")
      if tr.get("output_snippet"):
        snippet = tr["output_snippet"].strip().splitlines()
        print("Output snippet:", snippet[0][:200])
  else:
    print(out)


def main():
  load_dotenv()

  parser = argparse.ArgumentParser(
      prog="repo-agent",
      description="LLM-powered repository agent for code fixing and evaluation.",
  )
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # -------------------------------------------------------------------------
  # run: Run agent on a single task
  # -------------------------------------------------------------------------
  run_parser = subparsers.add_parser("run", help="Run agent on a single repo/goal")
  run_parser.add_argument("--repo", type=str, required=True, help="Path to target repo")
  run_parser.add_argument("--goal", type=str, required=True, help="What you want the agent to do")
  run_parser.add_argument("--trace", type=str, default="runs/trace.jsonl")
  run_parser.add_argument("--test", type=str, default="", help='Test command, e.g. "python -m pytest -q"')
  run_parser.add_argument(
      "--llm-provider",
      type=str,
      choices=["openai", "together"],
      default="openai",
      help="LLM provider backend (default: openai).",
  )
  run_parser.add_argument("--model", type=str, default=None, help="Model to use (overrides provider default).")
  run_parser.add_argument("--together-api-key", type=str, default=None, help="Together API key override.")
  run_parser.add_argument("--sandbox", dest="sandbox", action=argparse.BooleanOptionalAction, default=True,
                          help="Run against a temporary sandbox copy of the repo (default: enabled).")
  run_parser.add_argument("--sandbox-dir", type=str, default=None, help="Optional explicit sandbox directory to use.")
  run_parser.add_argument("--keep-sandbox", action="store_true", help="Keep the sandbox directory after the run.")
  run_parser.add_argument(
      "--test-policy",
      type=str,
      choices=["on_write", "on_final", "never"],
      default="on_write",
      help="When to run tests: after each write (on_write), only before finishing (on_final), or never.",
  )
  run_parser.set_defaults(func=cmd_run)

  # -------------------------------------------------------------------------
  # eval: Run evaluation suite
  # -------------------------------------------------------------------------
  eval_parser = subparsers.add_parser("eval", help="Run an evaluation suite")
  eval_parser.add_argument("--suite", type=str, required=True, help="Path to suite JSON file")
  eval_parser.add_argument("--trace-dir", type=str, default="runs/eval", help="Directory for trace files")
  eval_parser.add_argument("--report", type=str, default="runs/eval/report.json", help="Path for output report JSON")
  eval_parser.add_argument("--sandbox", dest="sandbox", action=argparse.BooleanOptionalAction, default=True,
                           help="Run tasks in sandbox mode (default: enabled).")
  eval_parser.add_argument("--keep-sandbox", action="store_true", help="Keep sandbox directories after runs.")
  eval_parser.add_argument(
      "--test-policy",
      type=str,
      choices=["on_write", "on_final", "never"],
      default="on_write",
      help="When to run tests.",
  )
  eval_parser.add_argument("--max-iters", type=int, default=20, help="Max agent iterations per task")
  eval_parser.add_argument("--model", type=str, default=None, help="Model to use (overrides OPENAI_MODEL env)")
  eval_parser.add_argument(
      "--llm-provider",
      type=str,
      choices=["openai", "together"],
      default="openai",
      help="LLM provider backend (default: openai).",
  )
  eval_parser.add_argument("--together-api-key", type=str, default=None, help="Together API key override.")
  eval_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-task progress output")
  eval_parser.set_defaults(func=cmd_eval)

  # -------------------------------------------------------------------------
  # prefs: Generate preference data for DPO finetuning
  # -------------------------------------------------------------------------
  prefs_parser = subparsers.add_parser("prefs", help="Generate preference data for DPO finetuning")
  prefs_parser.add_argument("--suite", type=str, required=True, help="Path to suite JSON file")
  prefs_parser.add_argument("--rollouts", type=int, default=4, help="Number of rollouts per task (default: 4)")
  prefs_parser.add_argument("--out", type=str, default="runs/prefs/dpo_dataset.jsonl",
                            help="Output path for preference JSONL file")
  prefs_parser.add_argument(
      "--data-write-mode",
      type=str,
      choices=["overwrite", "append"],
      default="overwrite",
      help="Whether to overwrite or append to output files (default: overwrite).",
  )
  prefs_parser.add_argument("--trace-dir", type=str, default="runs/prefs", help="Directory for trace files")
  prefs_parser.add_argument(
      "--llm-provider",
      type=str,
      choices=["openai", "together"],
      default="together",
      help="LLM provider backend (default: together).",
  )
  prefs_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo",
                            help="Model to use (default: Qwen/Qwen2.5-72B-Instruct-Turbo)")
  prefs_parser.add_argument("--together-api-key", type=str, default=None, help="Together API key override.")
  prefs_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
  prefs_parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility (default: 42)")
  prefs_parser.add_argument("--max-iters", type=int, default=20, help="Max agent iterations per task (default: 20)")
  prefs_parser.add_argument(
      "--test-policy",
      type=str,
      choices=["on_write", "on_final", "never"],
      default="on_write",
      help="When to run tests (default: on_write).",
  )
  prefs_parser.add_argument("--sandbox", dest="sandbox", action=argparse.BooleanOptionalAction, default=True,
                            help="Run tasks in sandbox mode (default: enabled).")
  prefs_parser.add_argument("--keep-sandbox", action="store_true", help="Keep sandbox directories after runs.")
  prefs_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-task progress output")
  prefs_parser.set_defaults(func=cmd_prefs)

  # -------------------------------------------------------------------------
  # Parse and dispatch
  # -------------------------------------------------------------------------
  args = parser.parse_args()

  # Handle no subcommand (backward compatibility: treat as 'run' if --repo and --goal provided)
  if args.command is None:
    # Check if legacy args are present
    if hasattr(args, "repo") and hasattr(args, "goal"):
      cmd_run(args)
    else:
      parser.print_help()
      return

  # Dispatch to subcommand
  if hasattr(args, "func"):
    args.func(args)
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
