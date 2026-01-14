from __future__ import annotations
import argparse
import json
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
import llm_repo_agent.estimate_cost as estimate_cost
import llm_repo_agent.prefs.rollouts as prefs_rollouts
import llm_repo_agent.sft.config as sft_config
import llm_repo_agent.sft.extract as sft_extract
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
      tool_protocol=args.tool_protocol,
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
      tool_protocol=args.tool_protocol,
      progress=not args.quiet,
  )

  runner = eval_runner.EvalRunner(cfg=cfg)
  num_workers = getattr(args, "num_workers", 0) or 0
  if num_workers > 1:
    results = runner.run_suite_parallel(suite, max_workers=num_workers)
  else:
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
      max_workers=args.max_workers,
      sandbox=args.sandbox,
      keep_sandbox=args.keep_sandbox,
      test_policy=args.test_policy,
      max_iters=args.max_iters,
      model=args.model,
      llm_provider=args.llm_provider,
      together_api_key=args.together_api_key,
      tool_protocol=args.tool_protocol,
      temperature=args.temperature,
      base_seed=args.seed,
      progress=not args.quiet,
  )

  prefs_rollouts.run_rollouts(Path(args.suite), cfg)


def cmd_estimate_cost(args):
  """Estimate preference data generation cost from trace logs."""
  trace_dir = Path(args.trace_dir)
  dataset_path = Path(args.dataset)

  usage = estimate_cost.collect_usage_stats(trace_dir)
  if usage.calls == 0:
    print(f"[estimate-cost] No llm_usage events found in: {trace_dir}")
    return

  if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

  pair_count = estimate_cost.count_pairs(dataset_path)
  target_pairs = args.target_pairs if args.target_pairs is not None else pair_count

  result = estimate_cost.estimate_cost(
      usage=usage,
      pair_count=pair_count,
      price_in=args.price_in,
      price_out=args.price_out,
      target_pairs=target_pairs,
  )

  if args.json:
    print(json.dumps(result, indent=2))
    return

  print(f"[estimate-cost] calls={result['calls']} pairs={result['pairs']} files_scanned={result['files_scanned']}")
  print(f"[estimate-cost] avg_prompt_tokens={result['avg_prompt_tokens']:.1f} avg_completion_tokens={result['avg_completion_tokens']:.1f}")
  print(f"[estimate-cost] cost_per_call=${result['cost_per_call']:.6f}")
  print(f"[estimate-cost] calls_per_pair={result['calls_per_pair']:.2f} cost_per_pair=${result['cost_per_pair']:.6f}")
  print(f"[estimate-cost] target_pairs={result['target_pairs']} scaled_cost=${result['scaled_cost']:.2f}")


def cmd_sft_extract(args):
  """Extract step-level SFT data from trace logs."""
  cfg = sft_config.SFTExtractConfig(
      trace_dir=Path(args.trace_dir),
      output_path=Path(args.output),
      require_success=args.require_success,
      require_valid_tool_ok=args.require_valid_tool_ok,
      drop_postfix_on_loop=args.drop_postfix_on_loop,
      filter_write_file_targets=args.filter_write_file_targets,
      require_root_list_files_first=args.require_root_list_files_first,
      max_context_chars=args.max_context_chars,
      output_format=args.output_format,
      progress=not args.quiet,
  )

  samples = sft_extract.extract_sft_samples(cfg)
  cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
  with cfg.output_path.open("w", encoding="utf-8") as f:
    for sample in samples:
      f.write(json.dumps(sample, ensure_ascii=False) + "\n")

  print(f"[sft-extract] Wrote {len(samples)} samples to {cfg.output_path}")


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
  run_parser.add_argument(
      "--tool-protocol",
      "--tool-mode",
      dest="tool_protocol",
      type=str,
      choices=["native", "json"],
      default="native",
      help="Tool calling protocol (native tool calls or JSON-in-content).",
  )
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
      "--tool-protocol",
      "--tool-mode",
      dest="tool_protocol",
      type=str,
      choices=["native", "json"],
      default="native",
      help="Tool calling protocol (native tool calls or JSON-in-content).",
  )
  eval_parser.add_argument(
      "--llm-provider",
      type=str,
      choices=["openai", "together"],
      default="openai",
      help="LLM provider backend (default: openai).",
  )
  eval_parser.add_argument(
      "--num-workers",
      "--num-worker",
      dest="num_workers",
      type=int,
      default=0,
      help="Number of parallel worker threads for eval (0/1 = sequential).",
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
  prefs_parser.add_argument("--max-workers", type=int, default=4,
                            help="Max parallel rollouts (0 for sequential, default: 4)")
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
  prefs_parser.add_argument(
      "--tool-protocol",
      type=str,
      choices=["native", "json"],
      default="native",
      help="Tool calling protocol (native tool calls or JSON-in-content).",
  )
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
  # sft-extract: Extract SFT training data from trace logs
  # -------------------------------------------------------------------------
  sft_parser = subparsers.add_parser("sft-extract", help="Extract SFT training data from trace logs")
  sft_parser.add_argument("--trace-dir", type=str, required=True, help="Directory containing trace JSONL files")
  sft_parser.add_argument("--output", type=str, required=True, help="Output JSONL file for SFT dataset")
  sft_parser.add_argument("--require-success", action="store_true", default=True,
                          help="Only include runs where tests passed (default: True)")
  sft_parser.add_argument("--no-require-success", dest="require_success", action="store_false",
                          help="Include runs regardless of test outcome")
  sft_parser.add_argument("--require-valid-tool-ok", action="store_true", default=True,
                          help="Only include steps where tool_result.ok is True (default: True)")
  sft_parser.add_argument("--no-require-valid-tool-ok", dest="require_valid_tool_ok", action="store_false",
                          help="Include steps regardless of tool_result.ok")
  sft_parser.add_argument(
      "--drop-postfix-on-loop",
      "--drop-post-fix-on-loop",
      dest="drop_postfix_on_loop",
      action="store_true",
      default=False,
      help="Stop emitting samples after a loop is detected (default: False)",
  )
  sft_parser.add_argument(
      "--filter-write-file-targets",
      action="store_true",
      default=False,
      help="Drop write_file samples that target test files or non-goal paths (default: False)",
  )
  sft_parser.add_argument(
      "--require-root-list-files-first",
      action="store_true",
      default=False,
      help="Only keep runs where the first tool call is list_files on '.' (default: False)",
  )
  sft_parser.add_argument("--max-context-chars", type=int, default=8000,
                          help="Max chars for tool output in context (default: 8000)")
  sft_parser.add_argument(
      "--format",
      dest="output_format",
      type=str,
      choices=["json", "native"],
      default="json",
      help="Output format for SFT samples (default: json).",
  )
  sft_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
  sft_parser.set_defaults(func=cmd_sft_extract)

  # -------------------------------------------------------------------------
  # estimate-cost: Estimate preference data generation cost
  # -------------------------------------------------------------------------
  cost_parser = subparsers.add_parser("estimate-cost", help="Estimate preference data generation cost")
  cost_parser.add_argument("--trace-dir", type=str, required=True, help="Directory containing trace JSONL files")
  cost_parser.add_argument("--dataset", type=str, required=True, help="Preference dataset JSONL file")
  cost_parser.add_argument("--price-in", type=float, required=True, help="Input token price per 1M tokens (USD)")
  cost_parser.add_argument("--price-out", type=float, required=True, help="Output token price per 1M tokens (USD)")
  cost_parser.add_argument("--target-pairs", type=int, default=None, help="Target number of pairs to scale to")
  cost_parser.add_argument("--json", action="store_true", help="Print JSON output")
  cost_parser.set_defaults(func=cmd_estimate_cost)

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
