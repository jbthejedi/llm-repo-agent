"""Run N rollouts per task for preference data generation."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_repo_agent.eval.runner import EvalConfig, EvalRunner, TaskResult
from llm_repo_agent.eval.tasks import TaskSpec, EvalSuite, load_suite
from llm_repo_agent.llm import LLM, LLMFactory, LLMConfig
from llm_repo_agent.prompts import system_prompt

from .schema import PreferencePair, PreferenceMeta, format_together_jsonl
from .score import score_rollout, RolloutScore
from .pairs import select_pair, PairResult


@dataclass
class RolloutResult:
    """Result from running a single rollout."""
    task_result: TaskResult
    score: RolloutScore
    final_content: str  # JSON-serialized final output
    seed: int


@dataclass
class RolloutWorkItem:
    """A single unit of work for parallel execution."""
    task: TaskSpec
    task_idx: int  # Index in suite.tasks (for grouping when task_ids aren't unique)
    suite_name: str
    seed: int
    rollout_idx: int  # For progress reporting


@dataclass
class PrefsConfig:
    """Configuration for preference data generation.

    Attributes:
        trace_dir: Directory to store trace files.
        out_path: Path for output JSONL file.
        meta_path: Path for metadata JSONL file.
        write_mode: Output write mode ("overwrite" or "append").
        rollouts: Number of rollouts per task.
        max_workers: Number of parallel threads (0 = sequential).
        sandbox: Whether to run in sandbox mode.
        keep_sandbox: Whether to keep sandbox after run.
        test_policy: When to run tests.
        max_iters: Maximum agent iterations per task.
        model: Model identifier to use.
        llm_provider: Provider backend.
        together_api_key: Optional Together API key override.
        tool_protocol: Tool calling protocol ("native" or "json").
        temperature: Sampling temperature.
        base_seed: Base seed for reproducibility.
        progress: Whether to show progress output.
    """
    trace_dir: Path = field(default_factory=lambda: Path("runs/prefs"))
    out_path: Path = field(default_factory=lambda: Path("runs/prefs/dpo_dataset.jsonl"))
    meta_path: Optional[Path] = None  # Defaults to out_path with _meta suffix
    write_mode: str = "overwrite"  # "overwrite" | "append"
    rollouts: int = 4
    max_workers: int = 4  # Number of parallel threads (0 = sequential)
    sandbox: bool = True
    keep_sandbox: bool = False
    test_policy: str = "on_write"
    max_iters: int = 20
    model: Optional[str] = None
    llm_provider: str = "together"
    together_api_key: Optional[str] = None
    tool_protocol: str = "native"
    temperature: float = 0.7
    base_seed: int = 42
    progress: bool = True

    def __post_init__(self):
        if self.write_mode not in {"overwrite", "append"}:
            raise ValueError(f"Invalid write_mode: {self.write_mode!r}. Use 'overwrite' or 'append'.")
        if self.meta_path is None:
            stem = self.out_path.stem
            self.meta_path = self.out_path.parent / f"{stem}_meta.jsonl"


def run_single_rollout(item: RolloutWorkItem, cfg: PrefsConfig) -> RolloutResult:
    """Execute a single rollout. Thread-safe, no shared mutable state.

    Args:
        item: Work item containing task and seed.
        cfg: Configuration (read-only).

    Returns:
        RolloutResult with task_result, score, final_content, seed.
    """
    # Create eval config for this rollout
    eval_cfg = EvalConfig(
        trace_dir=cfg.trace_dir,
        sandbox=cfg.sandbox,
        keep_sandbox=cfg.keep_sandbox,
        test_policy=cfg.test_policy,
        max_iters=cfg.max_iters,
        model=cfg.model,
        llm_provider=cfg.llm_provider,
        together_api_key=cfg.together_api_key,
        tool_protocol=cfg.tool_protocol,
        print_mode="quiet",  # Disable per-step progress in parallel mode
    )

    # Create LLM factory with seed
    def llm_factory() -> LLM:
        return LLMFactory.build(LLMConfig(
            provider=cfg.llm_provider,
            model=cfg.model,
            together_api_key=cfg.together_api_key,
            temperature=cfg.temperature,
            seed=item.seed,
            tool_protocol=cfg.tool_protocol,
        ))

    # Run the task
    runner = EvalRunner(cfg=eval_cfg, llm_factory=llm_factory)
    task_result = runner.run_task(item.task)

    # Score it
    score = score_rollout(task_result)

    # Build final content JSON
    final_obj = {
        "type": "final",
        "summary": task_result.final_summary,
        "changes": [{"path": p, "description": "Edited file"} for p in task_result.files_touched],
    }
    if task_result.success is not None:
        final_obj["test_result"] = {
            "ok": task_result.success,
            "summary": task_result.test_output[:200] if task_result.test_output else "",
        }

    return RolloutResult(
        task_result=task_result,
        score=score,
        final_content=json.dumps(final_obj),
        seed=item.seed,
    )


class PrefsRunner:
    """Runs rollouts and generates preference pairs."""

    def __init__(self, cfg: PrefsConfig):
        self.cfg = cfg
        self.pairs: List[PreferencePair] = []
        self.metas: List[PreferenceMeta] = []
        self.no_contrast_tasks: List[str] = []
        # Threading support
        self._progress_lock = threading.Lock()
        self._completed_count = 0
        self._total_work_items = 0

    def _make_llm_factory(self, seed: int) -> Callable[[], LLM]:
        """Create an LLM factory with a specific seed."""
        def factory() -> LLM:
            return LLMFactory.build(LLMConfig(
                provider=self.cfg.llm_provider,
                model=self.cfg.model,
                together_api_key=self.cfg.together_api_key,
                temperature=self.cfg.temperature,
                seed=seed,
                tool_protocol=self.cfg.tool_protocol,
            ))
        return factory

    def _report_progress(self, item: RolloutWorkItem, result: RolloutResult) -> None:
        """Thread-safe progress reporting."""
        if not self.cfg.progress:
            return
        with self._progress_lock:
            self._completed_count += 1
            status = "PASS" if result.task_result.success else (
                "FAIL" if result.task_result.success is False else "N/A"
            )

            # Determine failure reason for FAIL/N/A cases
            reason = ""
            if status in ("FAIL", "N/A"):
                tr = result.task_result
                if tr.error:
                    err_lower = tr.error.lower()
                    if "rate" in err_lower or "limit" in err_lower or "429" in err_lower:
                        reason = " rate_limit"
                    elif "parse" in err_lower or "json" in err_lower:
                        reason = " json_parse_error"
                    else:
                        # Truncate error message for display
                        reason = f" error:{tr.error[:40]}"
                elif tr.steps >= self.cfg.max_iters:
                    reason = " max_iters_reached"
                elif tr.parse_errors > 0:
                    reason = f" parse_errors:{tr.parse_errors}"
                elif status == "FAIL" and tr.test_output:
                    # Test failure - show first line of output
                    first_line = tr.test_output.strip().split('\n')[0][:40]
                    reason = f" test_fail:{first_line}"

            print(f"[prefs] [{self._completed_count}/{self._total_work_items}] "
                  f"{item.task.task_id} seed={item.seed}: {status}{reason} "
                  f"(steps={result.task_result.steps})")

    def run_task_rollouts(self, task: TaskSpec, suite_name: str) -> Optional[tuple[PreferencePair, PreferenceMeta]]:
        """Run N rollouts for a single task and return preference pair if contrast exists.

        Args:
            task: The task specification.
            suite_name: Name of the suite (for metadata).

        Returns:
            Tuple of (PreferencePair, PreferenceMeta) if contrast exists, None otherwise.
        """
        rollout_results: List[RolloutResult] = []

        for i in range(self.cfg.rollouts):
            seed = self.cfg.base_seed + i

            if self.cfg.progress:
                print(f"\n[prefs] Task {task.task_id}, rollout {i+1}/{self.cfg.rollouts} (seed={seed})")

            # Create eval config for this rollout
            eval_cfg = EvalConfig(
                trace_dir=self.cfg.trace_dir,
                sandbox=self.cfg.sandbox,
                keep_sandbox=self.cfg.keep_sandbox,
                test_policy=self.cfg.test_policy,
                max_iters=self.cfg.max_iters,
                model=self.cfg.model,
                llm_provider=self.cfg.llm_provider,
                together_api_key=self.cfg.together_api_key,
                print_mode="verbose" if self.cfg.progress else "quiet",
            )

            # Create runner with seed-specific LLM factory
            runner = EvalRunner(cfg=eval_cfg, llm_factory=self._make_llm_factory(seed))
            task_result = runner.run_task(task)

            # Score the rollout
            score = score_rollout(task_result)

            # Extract final content as JSON
            final_obj = {
                "type": "final",
                "summary": task_result.final_summary,
                "changes": [{"path": p, "description": "Edited file"} for p in task_result.files_touched],
            }
            if task_result.success is not None:
                final_obj["test_result"] = {
                    "ok": task_result.success,
                    "summary": task_result.test_output[:200] if task_result.test_output else "",
                }
            final_content = json.dumps(final_obj)

            rollout_results.append(RolloutResult(
                task_result=task_result,
                score=score,
                final_content=final_content,
                seed=seed,
            ))

            if self.cfg.progress:
                status = "PASS" if task_result.success else ("FAIL" if task_result.success is False else "N/A")
                print(f"[prefs] Rollout {i+1} result: {status}, steps={task_result.steps}, tool_calls={task_result.tool_calls}")

        # Select best/worst pair
        scores = [r.score for r in rollout_results]
        pair_result = select_pair(scores)

        if pair_result is None or not pair_result.has_contrast:
            if self.cfg.progress:
                print(f"[prefs] No contrast for task {task.task_id} - skipping")
            self.no_contrast_tasks.append(task.task_id)
            return None

        preferred = rollout_results[pair_result.preferred_idx]
        non_preferred = rollout_results[pair_result.non_preferred_idx]

        # Build preference pair
        sys_prompt = system_prompt()
        user_goal = f"GOAL:\n{task.goal}"

        pref_pair = format_together_jsonl(
            system_prompt=sys_prompt,
            user_goal=user_goal,
            preferred_content=preferred.final_content,
            non_preferred_content=non_preferred.final_content,
        )

        # Build metadata
        meta = PreferenceMeta(
            task_id=task.task_id,
            suite=suite_name,
            model=self.cfg.model or "unknown",
            temperature=self.cfg.temperature,
            seed=self.cfg.base_seed,
            scores={
                "preferred": pair_result.preferred_score.primary,
                "non_preferred": pair_result.non_preferred_score.primary,
            },
            tests_ok={
                "preferred": preferred.task_result.success or False,
                "non_preferred": non_preferred.task_result.success or False,
            },
            trace_ids={
                "preferred": preferred.task_result.run_id,
                "non_preferred": non_preferred.task_result.run_id,
            },
            rollout_counts={"total": self.cfg.rollouts},
        )

        if self.cfg.progress:
            print(f"[prefs] Selected pair: preferred=rollout[{pair_result.preferred_idx}] (score={pair_result.preferred_score.primary}), "
                  f"non_preferred=rollout[{pair_result.non_preferred_idx}] (score={pair_result.non_preferred_score.primary})")

        return pref_pair, meta

    def _form_pair(
        self,
        task: TaskSpec,
        suite_name: str,
        rollout_results: List[RolloutResult]
    ) -> Optional[tuple[PreferencePair, PreferenceMeta]]:
        """Form a preference pair from rollout results."""
        scores = [r.score for r in rollout_results]
        pair_result = select_pair(scores)

        if pair_result is None or not pair_result.has_contrast:
            return None

        preferred = rollout_results[pair_result.preferred_idx]
        non_preferred = rollout_results[pair_result.non_preferred_idx]

        sys_prompt = system_prompt()
        user_goal = f"GOAL:\n{task.goal}"

        pref_pair = format_together_jsonl(
            system_prompt=sys_prompt,
            user_goal=user_goal,
            preferred_content=preferred.final_content,
            non_preferred_content=non_preferred.final_content,
        )

        meta = PreferenceMeta(
            task_id=task.task_id,
            suite=suite_name,
            model=self.cfg.model or "unknown",
            temperature=self.cfg.temperature,
            seed=self.cfg.base_seed,
            scores={
                "preferred": pair_result.preferred_score.primary,
                "non_preferred": pair_result.non_preferred_score.primary,
            },
            tests_ok={
                "preferred": preferred.task_result.success or False,
                "non_preferred": non_preferred.task_result.success or False,
            },
            trace_ids={
                "preferred": preferred.task_result.run_id,
                "non_preferred": non_preferred.task_result.run_id,
            },
            rollout_counts={"total": self.cfg.rollouts},
        )

        return pref_pair, meta

    def run_suite_parallel(self, suite: EvalSuite) -> None:
        """Run all rollouts in parallel, then group and form pairs."""
        self.pairs = []
        self.metas = []
        self.no_contrast_tasks = []

        # Build flat list of all work items
        work_items: List[RolloutWorkItem] = []
        for task_idx, task in enumerate(suite.tasks):
            for i in range(self.cfg.rollouts):
                work_items.append(RolloutWorkItem(
                    task=task,
                    task_idx=task_idx,
                    suite_name=suite.name,
                    seed=self.cfg.base_seed + i,
                    rollout_idx=i,
                ))

        self._total_work_items = len(work_items)
        self._completed_count = 0

        if self.cfg.progress:
            print(f"[prefs] Starting {self._total_work_items} rollouts "
                  f"({len(suite.tasks)} tasks × {self.cfg.rollouts} rollouts) "
                  f"with {self.cfg.max_workers} workers")

        # Execute all in parallel
        # Use task_idx as key to handle duplicate task_ids in suite
        results_by_task_idx: Dict[int, List[RolloutResult]] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            # Submit all work items
            future_to_item = {
                executor.submit(run_single_rollout, item, self.cfg): item
                for item in work_items
            }

            # Collect results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results_by_task_idx[item.task_idx].append(result)
                    self._report_progress(item, result)
                except Exception as e:
                    print(f"[prefs] ERROR: {item.task.task_id} seed={item.seed}: {e}")
                    # Best-effort: log and continue, skip failed rollouts

        # Process each task's results to form pairs
        for task_idx, task in enumerate(suite.tasks):
            task_results = results_by_task_idx.get(task_idx, [])
            if len(task_results) < 2:
                self.no_contrast_tasks.append(task.task_id)
                continue

            pair_result = self._form_pair(task, suite.name, task_results)
            if pair_result:
                pref_pair, meta = pair_result
                self.pairs.append(pref_pair)
                self.metas.append(meta)
            else:
                self.no_contrast_tasks.append(task.task_id)

        # Print summary
        if self.cfg.progress:
            print(f"\n{'='*60}")
            print(f"[prefs] Summary:")
            print(f"[prefs] Total tasks: {len(suite.tasks)}")
            print(f"[prefs] Preference pairs generated: {len(self.pairs)}")
            print(f"[prefs] No-contrast tasks: {len(self.no_contrast_tasks)}")
            if self.no_contrast_tasks:
                print(f"[prefs] Skipped: {', '.join(self.no_contrast_tasks)}")
            print(f"{'='*60}")

    def run_suite(self, suite: EvalSuite) -> None:
        """Run suite - parallel if max_workers > 0, else sequential."""
        if self.cfg.max_workers > 0:
            self.run_suite_parallel(suite)
        else:
            self._run_suite_sequential(suite)

    def _run_suite_sequential(self, suite: EvalSuite) -> None:
        """Run all tasks in a suite sequentially and generate preference pairs.

        Args:
            suite: The evaluation suite to process.
        """
        self.pairs = []
        self.metas = []
        self.no_contrast_tasks = []

        for task in suite.tasks:
            if self.cfg.progress:
                print(f"\n{'='*60}")
                print(f"[prefs] Processing task: {task.task_id}")
                print(f"[prefs] Goal: {task.goal}")
                print(f"{'='*60}")

            result = self.run_task_rollouts(task, suite.name)
            if result is not None:
                pref_pair, meta = result
                self.pairs.append(pref_pair)
                self.metas.append(meta)

        if self.cfg.progress:
            print(f"\n{'='*60}")
            print(f"[prefs] Summary:")
            print(f"[prefs] Total tasks: {len(suite.tasks)}")
            print(f"[prefs] Preference pairs generated: {len(self.pairs)}")
            print(f"[prefs] No-contrast tasks: {len(self.no_contrast_tasks)}")
            if self.no_contrast_tasks:
                print(f"[prefs] Skipped: {', '.join(self.no_contrast_tasks)}")
            print(f"{'='*60}")

    def write_output(self) -> None:
        """Write preference pairs and metadata to JSONL files."""
        # Ensure output directories exist
        self.cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.cfg.write_mode == "append" else "w"

        def _ensure_newline(path: Path) -> None:
            if self.cfg.write_mode != "append":
                return
            if not path.exists():
                return
            if path.stat().st_size == 0:
                return
            with path.open("rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    with path.open("a", encoding="utf-8") as f_text:
                        f_text.write("\n")

        # Write preference pairs
        _ensure_newline(self.cfg.out_path)
        with self.cfg.out_path.open(mode, encoding="utf-8") as f:
            for pair in self.pairs:
                f.write(pair.to_jsonl() + "\n")

        if self.cfg.progress:
            verb = "Appended" if self.cfg.write_mode == "append" else "Wrote"
            print(f"[prefs] {verb} {len(self.pairs)} preference pairs to: {self.cfg.out_path}")

        # Write metadata
        if self.cfg.meta_path:
            _ensure_newline(self.cfg.meta_path)
            with self.cfg.meta_path.open(mode, encoding="utf-8") as f:
                for meta in self.metas:
                    f.write(meta.to_jsonl() + "\n")

            if self.cfg.progress:
                verb = "Appended" if self.cfg.write_mode == "append" else "Wrote"
                print(f"[prefs] {verb} metadata to: {self.cfg.meta_path}")


def run_rollouts(
    suite_path: Path,
    cfg: PrefsConfig,
) -> PrefsRunner:
    """Main entry point: load suite, run rollouts, write output.

    Args:
        suite_path: Path to the suite JSON file.
        cfg: Configuration for preference generation.

    Returns:
        PrefsRunner instance with results.
    """
    suite = load_suite(suite_path)

    if cfg.progress:
        print(f"[prefs] Loaded suite: {suite.name} ({len(suite.tasks)} tasks)")
        print(f"[prefs] Rollouts per task: {cfg.rollouts}")
        print(f"[prefs] Temperature: {cfg.temperature}")
        print(f"[prefs] Base seed: {cfg.base_seed}")

    runner = PrefsRunner(cfg)
    runner.run_suite(suite)
    runner.write_output()

    return runner
