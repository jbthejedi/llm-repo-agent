## 1) Repo-level RAG

**Summary:** Add a retrieval subsystem (chunk + index + retrieve) and expose it as a new tool `search_repo`, logging retrieval traces so we can evaluate retrieval separately from generation.

**What changes**

* **New module:** `llm_repo_agent/retrieval/`

  * `chunker.py` — splits repo into chunks (start simple: function-level or file blocks)
  * `index.py` — builds an index (embeddings + metadata)
  * `store.py` — persistent storage (FAISS / sqlite / JSONL + numpy)
  * `retrieve.py` — returns top-k chunks for a query
* **New tool:** `search_repo(query, top_k, filters)` → returns:

  * chunk text
  * file path + line ranges
  * `chunk_id`
  * score

**Where it wires in**

* **RepoTools** gains `search_repo(...)`
* **ActionController** dispatches `"search_repo"` like other tools
* **History/Trace** log retrieval events:

  * query + parameters
  * retrieved `chunk_id`s + scores
  * what context was injected into the model prompt

**Why this matters**

* You can answer “did the model fail because retrieval was bad, or because generation was bad?”—which is what serious teams care about.

---

## 2) Workspace sandbox

**Summary:** Run each attempt in an isolated working copy (git worktree / temp copy) so rollouts, patching, and tests are safe and reproducible.

**What changes**

* **New module:** `llm_repo_agent/workspace.py`

  * creates a disposable repo per run/attempt (best: `git worktree`, simplest: `copytree`)
  * manages cleanup + artifact retention (optional)

**Where it wires in**

* Driver (`RepoAgent.run`) starts by creating a workspace root.
* RepoTools takes `root_dir=workspace_root` (instead of operating on your original repo).

**Why this matters**

* For DPO you’ll generate **multiple candidates per task**. You need to apply/undo safely and run tests without contaminating state.

---

## 3) Eval harness

**Summary:** Add a suite runner that executes tasks, scores them (tests pass/fail), and emits a report + traces so you can measure progress across models and versions.

**What changes**

* **New module:** `llm_repo_agent/eval/`

  * `tasks.py` — `TaskSpec` (repo, goal, test_cmd, metadata)
  * `runner.py` — runs tasks, collects results
  * `metrics.py` — computes:

    * success rate (tests pass/fail)
    * avg steps / tool calls
    * retrieval stats (queries, top-k, context usage)
    * loop failures / reflection triggers
  * `report.py` — writes `report.json` + aggregates traces

**CLI changes**

* Turn `main.py` into subcommands:

  * `repo-agent index --repo ...`
  * `repo-agent run --repo ... --goal ... --test ...`
  * `repo-agent eval --suite eval/suites/my_suite.json`

**Output**

* Per-task trace JSONL (you already do this)
* Aggregate report JSON with baseline scores + deltas later

**Why this matters**

* This is the difference between a “toy agent demo” and a **measured research system**.

---

## 4) Preference data generation

**Summary:** Turn eval into training data by running multiple rollouts per task, scoring outcomes (tests), and converting them into (chosen, rejected) preference pairs for DPO.

**What changes**

* **New module:** `llm_repo_agent/prefs/`

  * `rollouts.py` — for each `TaskSpec`, run **N rollouts** with stochasticity (temperature, seeds)
  * `score.py` — score each rollout:

    * primary oracle: tests pass/fail
    * optional: lint/typecheck, smaller diff, fewer files touched
  * `pairs.py` — produce preference pairs:

    * chosen = best-scoring rollout
    * rejected = lower-scoring rollout(s)

**Dataset format**

* Write `dpo_dataset.jsonl` with:

  * `prompt` (task goal + minimal context)
  * `chosen` (winning patch/response)
  * `rejected` (losing patch/response)
  * `meta` (scores, test outputs summary, trace IDs, commit diff metadata)

**Why this matters**

* You’ve built the core of “RLHF for code” without needing human labelers: **tests become the preference signal**.

---

## 5) DPO training

**Summary:** Add a clean post-training pipeline that consumes `dpo_dataset.jsonl`, runs DPO (often via LoRA), produces a checkpoint/adapter, and then you re-run eval to show measurable improvement.

**What changes**

* **New module:** `llm_repo_agent/post_training/`

  * `train_dpo.py`

    * loads preference dataset
    * trains with DPO using a standard implementation (e.g., HF/TRL-style)
    * outputs a checkpoint or LoRA adapter

**LLM adapter changes**

* Add a local model backend alongside your API backend:

  * `HFLocalLLM` (loads local checkpoint/adapter)
  * optionally `VLLMEndpointLLM` (served model)
* CLI can select:

  * baseline: `--llm openai:...`
  * tuned: `--llm hf:./checkpoints/dpo_run_001`

**Why this matters**

* You can run “before vs after” eval and show deltas—this is what converts skepticism into respect.

---

## 6) Prompt + control-loop tweaks

**Summary:** Minimal prompt updates so the agent learns a reliable workflow (retrieve → inspect → patch → test), while preserving your driver invariants (one action/turn, step limits, full trace).

**What changes**

* Prompt updates:

  * teach the model `search_repo` exists + when to use it
  * encourage “retrieve before guessing” and “run tests before final”
* Control-loop remains strict:

  * one action per turn (your invariant)
  * step limit + stop conditions
  * “stuck” detection + reflection triggers (your existing ReflectionController)
* Logging stays first-class:

  * every tool call + observation
  * every retrieval injection
  * every test output summary

**Why this matters**

* Most agent demos fail because they’re un-debuggable. Your scaffold already prevents that—this step keeps it true as you add RAG + training.
