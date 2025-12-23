No fundamental revision. Your refactor actually *tightened the seams* we need (typed `Action`s, LLM adapter owns parsing, `History` as ledger, `Trace` as ground truth). The plan stays the same; we just change **where** we plug things in.

## What stays identical

* We still add: **(1) repo RAG tool**, **(2) workspace sandbox**, **(3) eval harness**, **(4) preference pair generator**, **(5) DPO trainer**, **(6) minimal prompt/loop tweaks**.
* The driver loop + logging architecture you already have is still the backbone.

## The only real adjustments (because of your refactor)

### A) “Add a tool” now means updating 4 explicit places

When we add `search_repo`, we must update all of:

1. `actions.py` → `TOOL_NAMES` allowlist
2. `llm.py` → `TOOLS` schema passed to Responses API
3. `prompts.py` → `TOOL_SPEC` + the “allowed tool names” text
4. `controller.py` + `tools.py` → dispatch + implementation

That’s not a plan change, just a clearer integration checklist.

### B) Strong recommendation: create a single source of truth for tool specs

Right now tool definitions are duplicated (`llm.py` has `TOOLS`, `prompts.py` has `TOOL_SPEC`, `actions.py` has `TOOL_NAMES`). As soon as we add `search_repo`, this can drift.

Small revision: add something like `tool_schema.py` and import from it in all three places:

* `TOOL_NAMES`
* OpenAI `TOOLS` JSON schema
* Human-readable `TOOL_SPEC` for the prompt (can be derived)

This keeps the RLHF/RAG expansion from turning into “update 3 files and forget one.”

### C) Tests should become policy-driven (tiny config tweak)

Today: agent auto-runs tests after `write_file` (driver-triggered inside the loop). That’s fine for interactive runs, but for eval + preference rollouts it can cause extra test runs.

Tiny revision: add a config flag like:

* `test_policy = "on_write" | "on_final" | "never"`

For eval/prefs you’ll likely want **`on_final`** (score the rollout once, cleanly). For normal usage, keep `on_write`.

### D) Workspace sandbox plugs in cleanly already

Because `RepoTools(repo_root)` is the only “where tools point,” we just make `repo_root` a workspace path. No other architectural changes needed.

---

## Net: do we revise the plan?

**No.** We keep the same plan, and your refactor just makes implementation more straightforward—especially because tool calling + typed actions + History/Trace are already clean.

The only “revision” I’d insist on is **centralizing tool specs**, because adding `search_repo` + later “apply_patch” or “open_chunk” will otherwise become death-by-duplication.
