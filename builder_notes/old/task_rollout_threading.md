# Parallel Rollout Execution with ThreadPoolExecutor

## Overview

Currently, preference data generation runs N tasks × M rollouts sequentially. Since LLM API calls are I/O-bound (network latency), we can parallelize all N×M rollouts using `ThreadPoolExecutor` for significant speedup.

## Current Flow (Sequential)

```
for task in suite.tasks:           # N tasks
    for i in range(rollouts):      # M rollouts per task
        run_single_rollout(...)    # Sequential execution
    select_pair(results)
```

**Total time**: N × M × avg_rollout_time

## Proposed Flow (Parallel)

```
work_items = [(task, seed) for task in tasks for seed in seeds]  # N×M items
results = ThreadPoolExecutor.map(run_single_rollout, work_items)
grouped = group_by_task(results)
for task_id, task_results in grouped:
    select_pair(task_results)
```

**Total time**: max(rollout_times) + overhead ≈ avg_rollout_time (with enough workers)

---

## Implementation Plan

### Step 1: Add `max_workers` to PrefsConfig

**File**: `src/llm_repo_agent/prefs/rollouts.py`

```python
@dataclass
class PrefsConfig:
    # ... existing fields ...
    max_workers: int = 4  # Number of parallel threads (0 = sequential)
```

**File**: `src/llm_repo_agent/main.py`

Add CLI argument:
```python
prefs_parser.add_argument(
    "--max-workers", type=int, default=4,
    help="Max parallel rollouts (0 for sequential, default: 4)"
)
```

---

### Step 2: Create WorkItem and extract single-rollout function

**File**: `src/llm_repo_agent/prefs/rollouts.py`

```python
@dataclass
class RolloutWorkItem:
    """A single unit of work for parallel execution."""
    task: TaskSpec
    suite_name: str
    seed: int
    rollout_idx: int  # For progress reporting


def run_single_rollout(item: RolloutWorkItem, cfg: PrefsConfig) -> RolloutResult:
    """Execute a single rollout. Thread-safe, no shared mutable state.

    Args:
        item: Work item containing task and seed
        cfg: Configuration (read-only)

    Returns:
        RolloutResult with task_result, score, final_content, seed
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
        progress=False,  # Disable per-step progress in parallel mode
    )

    # Create LLM factory with seed
    def llm_factory() -> LLM:
        return LLMFactory.build(LLMConfig(
            provider=cfg.llm_provider,
            model=cfg.model,
            together_api_key=cfg.together_api_key,
            temperature=cfg.temperature,
            seed=item.seed,
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
```

---

### Step 3: Add parallel execution to PrefsRunner

**File**: `src/llm_repo_agent/prefs/rollouts.py`

```python
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class PrefsRunner:
    def __init__(self, cfg: PrefsConfig):
        self.cfg = cfg
        self.pairs: List[PreferencePair] = []
        self.metas: List[PreferenceMeta] = []
        self.no_contrast_tasks: List[str] = []
        self._progress_lock = threading.Lock()  # For thread-safe progress output
        self._completed_count = 0
        self._total_work_items = 0

    def _report_progress(self, item: RolloutWorkItem, result: RolloutResult) -> None:
        """Thread-safe progress reporting."""
        if not self.cfg.progress:
            return
        with self._progress_lock:
            self._completed_count += 1
            status = "PASS" if result.task_result.success else (
                "FAIL" if result.task_result.success is False else "N/A"
            )
            print(f"[prefs] [{self._completed_count}/{self._total_work_items}] "
                  f"{item.task.task_id} seed={item.seed}: {status} "
                  f"(steps={result.task_result.steps})")

    def run_suite_parallel(self, suite: EvalSuite) -> None:
        """Run all rollouts in parallel, then group and form pairs."""
        self.pairs = []
        self.metas = []
        self.no_contrast_tasks = []

        # Build flat list of all work items
        work_items: List[RolloutWorkItem] = []
        for task in suite.tasks:
            for i in range(self.cfg.rollouts):
                work_items.append(RolloutWorkItem(
                    task=task,
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
        results_by_task: Dict[str, List[RolloutResult]] = defaultdict(list)

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
                    results_by_task[item.task.task_id].append(result)
                    self._report_progress(item, result)
                except Exception as e:
                    print(f"[prefs] ERROR: {item.task.task_id} seed={item.seed}: {e}")
                    # Create a failed result
                    # (or re-raise depending on error handling strategy)

        # Process each task's results to form pairs
        for task in suite.tasks:
            task_results = results_by_task.get(task.task_id, [])
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

    def run_suite(self, suite: EvalSuite) -> None:
        """Run suite - parallel if max_workers > 0, else sequential."""
        if self.cfg.max_workers > 0:
            self.run_suite_parallel(suite)
        else:
            self._run_suite_sequential(suite)

    def _run_suite_sequential(self, suite: EvalSuite) -> None:
        """Original sequential implementation (for max_workers=0)."""
        # ... keep existing run_suite logic here, renamed ...
```

---

### Step 4: Update main.py

**File**: `src/llm_repo_agent/main.py`

```python
def cmd_prefs(args):
    cfg = prefs_rollouts.PrefsConfig(
        # ... existing fields ...
        max_workers=args.max_workers,
    )
    prefs_rollouts.run_rollouts(Path(args.suite), cfg)

# In argument parser:
prefs_parser.add_argument(
    "--max-workers", type=int, default=4,
    help="Max parallel rollouts (0 for sequential, default: 4)"
)
```

---

## Thread Safety Considerations

1. **EvalRunner**: Each rollout creates its own instance - no shared state
2. **Sandbox**: Each rollout creates its own sandbox directory - independent
3. **Trace files**: Each rollout writes to a unique trace file (run_id-based) - no conflict
4. **LLM client**: OpenAI/Together clients are thread-safe for concurrent requests
5. **Progress output**: Protected by `threading.Lock()` for atomic prints

---

## Error Handling Strategy

Option A (Fail-fast): Re-raise exceptions, abort all pending work
Option B (Best-effort): Log errors, continue with other rollouts, skip failed tasks

**Recommendation**: Option B for robustness - one failed rollout shouldn't abort hours of work.
<question>
Q: if we get rate limiting response from the requests we're making, how do we handle that?
</question>
<answer>
Implement retry with exponential backoff + jitter on 429/503 responses (respect `Retry-After` header if present). Centralize this in the LLM adapter (`TogetherLLM.complete()`) so both action and reflection calls share the behavior.

**Implementation approach:**
1. Catch 429/503 in the LLM adapter
2. Use exponential backoff: `delay = min(base * 2^attempt + random_jitter, max_delay)`
3. Add optional CLI knobs: `--max-retries` (default: 3), `--backoff-max` (default: 60s)
4. When retries exhausted, mark the rollout as failed (best-effort) rather than aborting the whole job

**If rate limiting is frequent:**
- Lower `--max-workers` to reduce concurrent requests
- Or add a global `asyncio.Semaphore` / `threading.Semaphore` to cap concurrent LLM calls below the thread count

**Note:** Together's Python client (`together.Complete.create()`) may already have built-in retry logic. Check the client docs before implementing custom retry.
</answer>

---

## Rate Limiting

Together API has rate limits. Options:

1. **Start conservative**: Default `max_workers=4`
2. **Add CLI flag**: `--max-workers` lets users tune based on their API tier
3. **Future**: Add adaptive rate limiting with backoff

---

## Testing

1. Run with `--max-workers 1` - should behave like sequential
2. Run with `--max-workers 4` on small suite - verify same results
3. Run with `--max-workers 0` - should use sequential path
4. Verify trace files are written correctly (no corruption)
5. Verify progress output is readable (no interleaving)

---

## Files to Modify

1. `src/llm_repo_agent/prefs/rollouts.py` - Main changes
2. `src/llm_repo_agent/main.py` - Add `--max-workers` CLI arg
3. `tests/test_prefs_rollouts.py` - Add threading tests (optional)

---

## Example Usage

```bash
# Default: 4 parallel workers
poetry run repo-agent prefs --suite eval/suites/large_suite.json

# Conservative: 2 workers
poetry run repo-agent prefs --suite eval/suites/large_suite.json --max-workers 2

# Aggressive: 8 workers (if API tier supports)
poetry run repo-agent prefs --suite eval/suites/large_suite.json --max-workers 8

# Sequential (debugging):
poetry run repo-agent prefs --suite eval/suites/large_suite.json --max-workers 0
```
