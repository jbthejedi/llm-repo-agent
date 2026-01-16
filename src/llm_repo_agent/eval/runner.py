"""Evaluation runner that executes tasks and collects results."""

from __future__ import annotations

import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .tasks import TaskSpec, EvalSuite

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.llm import LLM, LLMFactory, LLMConfig
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace
from llm_repo_agent.sandbox import materialize_repo_sandbox, cleanup_sandbox, Sandbox


@dataclass
class TaskResult:
  """Result of running a single evaluation task.

    Attributes:
        task_id: ID of the task that was run.
        run_id: Unique run identifier for tracing.
        success: Whether tests passed (None if no tests or not run).
        steps: Number of agent iterations used.
        tool_calls: Number of tool calls made.
        files_touched: List of files modified during the run.
        error: Error message if the run failed unexpectedly.
        duration_s: Wall-clock time in seconds.
        final_summary: The agent's final summary.
        test_output: Raw test output (truncated).
        metadata: Any additional metadata from the task.
        reflection_count: Number of times reflection was triggered.
        loop_detections: Number of times loop tripwire fired.
        parse_errors: Number of times LLM returned malformed actions.
        test_runs: Number of test executions.
        tool_breakdown: Count of each tool used (e.g., {"read_file": 5, "write_file": 1}).
    """

  task_id: str
  run_id: str
  success: Optional[bool] = None
  steps: int = 0
  tool_calls: int = 0
  files_touched: List[str] = field(default_factory=list)
  error: Optional[str] = None
  duration_s: float = 0.0
  final_summary: str = ""
  test_output: str = ""
  metadata: Dict[str, Any] = field(default_factory=dict)
  reflection_count: int = 0
  loop_detections: int = 0
  parse_errors: int = 0
  test_runs: int = 0
  tool_breakdown: Dict[str, int] = field(default_factory=dict)
  valid_tool_actions: int = 0  # Successfully parsed tool_call actions

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class EvalConfig:
  """Configuration for the evaluation runner.

    Attributes:
        trace_dir: Directory to store trace files.
        sandbox: Whether to run in sandbox mode.
        keep_sandbox: Whether to keep sandbox after run.
        test_policy: When to run tests (on_write, on_final, never).
        max_iters: Maximum agent iterations per task.
        rollouts: Number of rollouts per task.
        temperature: Sampling temperature.
        base_seed: Base seed for reproducibility (None for random).
        model: Model identifier to use.
        llm_provider: Provider backend ("openai" or "together").
        together_api_key: Optional Together API key override.
        tool_protocol: Tool calling protocol ("native" or "json").
        print_mode: Output verbosity ("quiet", "standard", "verbose").
    """

  trace_dir: Path = field(default_factory=lambda: Path("runs/eval"))
  sandbox: bool = True
  keep_sandbox: bool = False
  test_policy: str = "on_write"
  max_iters: int = 20
  rollouts: int = 1
  temperature: float = 0.0
  base_seed: Optional[int] = None
  model: Optional[str] = None
  llm_provider: str = "openai"
  together_api_key: Optional[str] = None
  tool_protocol: str = "native"
  print_mode: str = "verbose"


class EvalRunner:
  """Runs evaluation tasks and collects results."""

  def __init__(
      self,
      cfg: Optional[EvalConfig] = None,
      llm_factory: Optional[Callable[[], LLM]] = None,
  ):
    """Initialize the evaluation runner.

        Args:
            cfg: Evaluation configuration.
            llm_factory: Optional factory function to create LLM instances.
                         If not provided, uses the configured provider via LLMFactory.
        """
    self.cfg = cfg or EvalConfig()
    self.llm_factory = llm_factory or self._default_llm_factory
    self.results: List[TaskResult] = []
    self._results_lock = threading.Lock()

  def _default_llm_factory(self) -> LLM:
    return LLMFactory.build(LLMConfig(
        provider=self.cfg.llm_provider,
        model=self.cfg.model,
        together_api_key=self.cfg.together_api_key,
        tool_protocol=self.cfg.tool_protocol,
    ))

  def run_task(self, task: TaskSpec) -> TaskResult:
    """Run a single evaluation task and return the result."""
    run_id = uuid.uuid4().hex[:10]
    start_time = time.time()

    task_result = TaskResult(
        task_id=task.task_id,
        run_id=run_id,
        metadata=task.metadata,
    )

    repo_root = Path(task.repo).expanduser().resolve()
    sandbox: Optional[Sandbox] = None
    tools_root = repo_root

    try:
      # Setup sandbox if enabled
      if self.cfg.sandbox:
        sandbox = materialize_repo_sandbox(repo_root)
        tools_root = sandbox.root

      # Setup tools and trace
      tools = RepoTools(repo_root=tools_root)
      trace_path = self.cfg.trace_dir / f"{task.task_id}_{run_id}.jsonl"
      llm = self.llm_factory()
      model_name = getattr(llm, "model", None) or (self.cfg.model or "unknown")
      if self.cfg.print_mode == "verbose":
        print(f"[llm] provider={self.cfg.llm_provider} model={model_name}")
      trace = Trace(
          trace_path,
          run_id=run_id,
          meta={
              "task_id": task.task_id,
              "repo": str(repo_root),
              "goal": task.goal,
              "test_cmd": task.test_cmd,
              "model": model_name,
              "provider": self.cfg.llm_provider,
          },
      )

      # Create agent
      agent_cfg = AgentConfig(
          max_iters=self.cfg.max_iters,
          test_policy=self.cfg.test_policy,
          progress=(self.cfg.print_mode == "verbose"),
      )
      agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=agent_cfg)

      # Run agent
      test_cmd = task.test_cmd_list()
      output = agent.run(goal=task.goal, test_cmd=test_cmd)

      # Extract results from output
      if isinstance(output, dict):
        task_result.final_summary = output.get("summary", "")
        task_result.files_touched = [c.get("path", "") for c in output.get("changes", [])]

        test_result = output.get("test_result")
        if test_result is not None:
          task_result.success = test_result.get("ok", False)
          task_result.test_output = test_result.get("output_snippet", "")[:2000]

      # Extract metrics from trace
      metrics = self._count_from_trace(trace, run_id)
      task_result.steps = metrics["steps"]
      task_result.tool_calls = metrics["tool_calls"]
      task_result.reflection_count = metrics["reflection_count"]
      task_result.loop_detections = metrics["loop_detections"]
      task_result.parse_errors = metrics["parse_errors"]
      task_result.test_runs = metrics["test_runs"]
      task_result.tool_breakdown = metrics["tool_breakdown"]
      task_result.valid_tool_actions = metrics["valid_tool_actions"]

    except Exception as e:
      task_result.error = str(e)
      task_result.success = False

    finally:
      # Cleanup sandbox
      if sandbox and not self.cfg.keep_sandbox:
        cleanup_sandbox(sandbox)

      task_result.duration_s = time.time() - start_time

    with self._results_lock:
      self.results.append(task_result)
    return task_result

  def _count_from_trace(self, trace: Trace, run_id: str) -> Dict[str, Any]:
    """Extract detailed metrics from trace events."""
    metrics = {
        "steps": 0,
        "tool_calls": 0,
        "reflection_count": 0,
        "loop_detections": 0,
        "parse_errors": 0,
        "test_runs": 0,
        "tool_breakdown": {},
        "valid_tool_actions": 0,
    }

    for evt in trace.iter_run_events(run_id):
      kind = evt.get("kind", "")
      payload = evt.get("payload", {})

      if kind == "llm_action":
        metrics["steps"] += 1
        # Count valid tool_call actions for instruction following metrics
        action = payload.get("action", {})
        if action.get("type") == "tool_call":
          metrics["valid_tool_actions"] += 1
      elif kind == "tool_result":
        metrics["tool_calls"] += 1
        tool_name = payload.get("tool", "")
        if tool_name:
          metrics["tool_breakdown"][tool_name] = metrics["tool_breakdown"].get(tool_name, 0) + 1
      elif kind == "reflection":
        metrics["reflection_count"] += 1
      elif kind == "driver_note":
        note = payload.get("note", "")
        if "Loop detected" in note:
          metrics["loop_detections"] += 1
      elif kind == "llm_parse_error":
        metrics["parse_errors"] += 1
      elif kind == "tests":
        metrics["test_runs"] += 1

    return metrics

  def run_suite(self, suite: EvalSuite) -> List[TaskResult]:
    """Run all tasks in an evaluation suite."""
    self.results = []

    for task in suite.tasks:
      if self.cfg.print_mode == "verbose":
        print(f"\n{'='*60}")
        print(f"[eval] Running task: {task.task_id}")
        print(f"[eval] Goal: {task.goal}")
        print(f"{'='*60}")

      task_result = self.run_task(task)

      if self.cfg.print_mode == "verbose":
        status = "PASS" if task_result.success else ("FAIL" if task_result.success is False else "N/A")
        print(f"\n[eval] Task {task.task_id}: {status}")
        print(f"[eval] Steps: {task_result.steps}, Tool calls: {task_result.tool_calls}")
        print(f"[eval] Duration: {task_result.duration_s:.1f}s")
        if task_result.error:
          print(f"[eval] Error: {task_result.error}")

    return self.results

  def run_suite_parallel(self, suite: EvalSuite, max_workers: int) -> List[TaskResult]:
    """Run all tasks in a suite using a thread pool."""
    self.results = []
    results_by_idx: Dict[int, TaskResult] = {}

    if self.cfg.print_mode == "verbose":
      print(f"[eval] Running in parallel with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {
        executor.submit(self.run_task, task): idx
        for idx, task in enumerate(suite.tasks)
      }
      for future in as_completed(futures):
        idx = futures[future]
        try:
          result = future.result()
        except Exception as exc:
          # Should be rare because run_task catches most errors.
          result = TaskResult(task_id=suite.tasks[idx].task_id, run_id="unknown", success=False, error=str(exc))
        results_by_idx[idx] = result
        if self.cfg.print_mode == "verbose":
          status = "PASS" if result.success else ("FAIL" if result.success is False else "N/A")
          print(f"\n[eval] Task {result.task_id}: {status}")
          print(f"[eval] Steps: {result.steps}, Tool calls: {result.tool_calls}")
          print(f"[eval] Duration: {result.duration_s:.1f}s")
          if result.error:
            print(f"[eval] Error: {result.error}")

    # Preserve original task order in results.
    self.results = [results_by_idx[i] for i in range(len(suite.tasks)) if i in results_by_idx]
    return self.results

  def run_tasks(self, tasks: List[TaskSpec]) -> List[TaskResult]:
    """Run a list of tasks (convenience method)."""
    suite = EvalSuite(name="adhoc", tasks=tasks)
    return self.run_suite(suite)

  def _create_llm_factory_with_seed(self, seed: int) -> Callable[[], LLM]:
    """Create an LLM factory with a specific seed."""
    def factory() -> LLM:
      return LLMFactory.build(LLMConfig(
          provider=self.cfg.llm_provider,
          model=self.cfg.model,
          together_api_key=self.cfg.together_api_key,
          tool_protocol=self.cfg.tool_protocol,
          temperature=self.cfg.temperature,
          seed=seed,
      ))
    return factory

  def run_task_with_seed(self, task: TaskSpec, seed: int) -> TaskResult:
    """Run a single task with a specific seed."""
    original_factory = self.llm_factory
    self.llm_factory = self._create_llm_factory_with_seed(seed)
    try:
      return self.run_task(task)
    finally:
      self.llm_factory = original_factory

  def _format_failure_reason(self, result: TaskResult) -> str:
    """Format failure reason for standard output (similar to prefs command)."""
    status = "PASS" if result.success else ("FAIL" if result.success is False else "N/A")
    if status == "PASS":
      return ""

    if result.error:
      err_lower = result.error.lower()
      if "rate" in err_lower or "limit" in err_lower or "429" in err_lower:
        return " rate_limit"
      elif "parse" in err_lower or "json" in err_lower:
        return " json_parse_error"
      else:
        return f" error:{result.error[:40]}"
    elif result.steps >= self.cfg.max_iters:
      return " max_iters_reached"
    elif result.parse_errors > 0:
      return f" parse_errors:{result.parse_errors}"
    elif status == "FAIL" and result.test_output:
      first_line = result.test_output.strip().split('\n')[0][:40]
      return f" test_fail:{first_line}"

    return ""

  def run_suite_with_rollouts(
      self,
      suite: EvalSuite,
      rollouts: int,
      max_workers: int = 0,
  ) -> "RolloutResults":
    """Run all tasks in a suite with multiple rollouts per task.

    Args:
        suite: Evaluation suite to run.
        rollouts: Number of rollouts per task.
        max_workers: Number of parallel threads (0 = sequential).

    Returns:
        RolloutResults containing all results grouped by task.
    """
    from collections import defaultdict

    task_results: Dict[str, List[TaskResult]] = defaultdict(list)
    print_mode = self.cfg.print_mode

    # Generate work items: (task, seed) pairs
    work_items = []
    for task_idx, task in enumerate(suite.tasks):
      for rollout_idx in range(rollouts):
        seed = (self.cfg.base_seed or 0) + task_idx * rollouts + rollout_idx
        work_items.append((task, seed, rollout_idx))

    if print_mode != "quiet":
      total = len(suite.tasks) * rollouts
      print(f"[eval] Running {len(suite.tasks)} tasks x {rollouts} rollouts = {total} total runs")

    if max_workers > 1:
      # Parallel execution
      completed = 0
      with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(self.run_task_with_seed, task, seed): (task, seed, rollout_idx)
            for task, seed, rollout_idx in work_items
        }
        for future in as_completed(futures):
          task, seed, rollout_idx = futures[future]
          try:
            result = future.result()
          except Exception as exc:
            result = TaskResult(
                task_id=task.task_id,
                run_id="unknown",
                success=False,
                error=str(exc),
                metadata=task.metadata,
            )
          task_results[task.task_id].append(result)
          completed += 1

          if print_mode == "verbose":
            status = "PASS" if result.success else ("FAIL" if result.success is False else "N/A")
            print(f"[eval] [{completed}/{len(work_items)}] {task.task_id} rollout {rollout_idx+1}: {status}")
          elif print_mode == "standard":
            status = "PASS" if result.success else ("FAIL" if result.success is False else "N/A")
            reason = self._format_failure_reason(result)
            print(f"[eval] [{completed}/{len(work_items)}] {task.task_id} seed={seed}: {status}{reason} (steps={result.steps})")
    else:
      # Sequential execution
      for i, (task, seed, rollout_idx) in enumerate(work_items):
        if print_mode == "verbose":
          print(f"\n[eval] [{i+1}/{len(work_items)}] Task: {task.task_id}, rollout {rollout_idx+1}")

        result = self.run_task_with_seed(task, seed)
        task_results[task.task_id].append(result)

        if print_mode == "verbose":
          status = "PASS" if result.success else ("FAIL" if result.success is False else "N/A")
          print(f"[eval] Result: {status}, Steps: {result.steps}, Duration: {result.duration_s:.1f}s")
        elif print_mode == "standard":
          status = "PASS" if result.success else ("FAIL" if result.success is False else "N/A")
          reason = self._format_failure_reason(result)
          print(f"[eval] [{i+1}/{len(work_items)}] {task.task_id} seed={seed}: {status}{reason} (steps={result.steps})")

    # Build flat results list for backward compatibility
    self.results = []
    for task in suite.tasks:
      self.results.extend(task_results[task.task_id])

    return RolloutResults(
        task_results=dict(task_results),
        rollouts_per_task=rollouts,
        total_tasks=len(suite.tasks),
    )


@dataclass
class RolloutResults:
  """Results from running a suite with multiple rollouts per task."""
  task_results: Dict[str, List[TaskResult]]  # task_id -> list of results
  rollouts_per_task: int
  total_tasks: int

  def get_task_pass_rate(self, task_id: str) -> float:
    """Get the pass rate for a specific task."""
    results = self.task_results.get(task_id, [])
    if not results:
      return 0.0
    passed = sum(1 for r in results if r.success is True)
    tested = sum(1 for r in results if r.success is not None and r.error is None)
    return passed / tested if tested > 0 else 0.0

  def get_task_summary(self, task_id: str) -> Dict[str, Any]:
    """Get a summary of results for a specific task."""
    results = self.task_results.get(task_id, [])
    passed = sum(1 for r in results if r.success is True)
    failed = sum(1 for r in results if r.success is False and r.error is None)
    errored = sum(1 for r in results if r.error)
    no_tests = sum(1 for r in results if r.success is None and r.error is None)
    total = len(results)
    tested = passed + failed
    return {
        "task_id": task_id,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "no_tests": no_tests,
        "total": total,
        "pass_rate": passed / tested if tested > 0 else 0.0,
    }

  def all_results(self) -> List[TaskResult]:
    """Get all results as a flat list."""
    results = []
    for task_results in self.task_results.values():
      results.extend(task_results)
    return results
