<h3><b>Draft write-up (Justin) + inline notes (Codex)</b></h3>

Okay, let's do a write-up before I embark on this project as any real scientist should do. My language will be loose and brainstormy. Correct anything that needs correcting

What's the problem we're trying to solve?

A smaller model like Qwen2.5-7B has multiple issues
1. it doesnt follow instructions well enough to make tool calls (--tool-protocol=json)
2. It's not intelligent enough to reason about the most optimal tool calls to make (I don't know this is try because the model can't make tool calls properly enough for me to see the chain of tool calls that it would make. But I'm pretty sure this is the case)
3. The model is not intelligent enough to suggest the correct patch fix for a certain difficulty level (unknown) of bug fix

> Note (Codex): it helps to split this into 2 separate problems so your training signal is clean:
> 1) **Tool-protocol compliance** (syntax + allowed tools + argument shapes). If the model can’t reliably emit valid JSON tool calls, you can’t even observe (2) or (3).
<question>
Question:
So then why don't we use SFT for tool-protocol compliance and not DPO? Is SFT not robust enough?

> Answer (Codex):
> - You *should* use **SFT for tool-protocol compliance**. It’s the most direct, cheapest way to teach “emit valid JSON with the right fields” and reduce `parse_errors`.
> - DPO isn’t “more robust” for syntax; it’s better for **behavioral preferences** when there are multiple plausible next actions (e.g., “read_file with enough context” vs “keep grepping”, “write_file now” vs “keep searching”, “stop and summarize” vs “thrash”).
>
> More concretely:
> - **SFT optimizes likelihood** of the target action. Great for: schema adherence, exact tool names, argument keys, consistent formatting.
> - **DPO optimizes relative ranking** between two actions. Great for: “prefer A over B” when both are valid but one is better policy.
>
> Practical recipe that’s common in agent/tool settings:
> > 1) SFT warm-start on “correct JSON tool calls” (even a few hundred–few thousand steps can help a lot).
> > 2) DPO on step-level pairs to shape: fewer loops, fewer wasted tool calls, earlier patching, better evidence usage.
>
> If you skip SFT and jump straight to DPO while the model can’t reliably output valid JSON, your “non-preferred” side will be dominated by parse failures and you won’t actually be training the policy you care about.
</question><br>

<question>
Question:
Ayayayay...So we're back to the SFT plan again. We were already here and I found that I couldn't generate enough SFT data (tool calls) to do a finetune. I feel like I'm going in circles

> Answer (Codex):
> - You’re not actually back at square one — the key unlock is: **SFT “samples” don’t have to be “unique tasks”**. For tool-protocol compliance, you can treat **each agent step** as one training example.
> - With step-level extraction, QuixBugs is enough to get to 1K–3K examples:
> > - ~50 Python tasks × 8 rollouts × ~10–15 usable steps ≈ **4K–6K** candidate SFT samples (before filtering).
>
> Why you previously “couldn’t generate enough”:
> - If you were counting “one task = one sample”, QuixBugs is too small.
> - But for teaching JSON tool calling, you mainly need lots of repetitions of: `list_files → grep/read_file → write_file` under different tool results.
>
> Practical scoped plan:
> > 1) Run a **strong teacher** on QuixBugs with `--tool-protocol=json` for many rollouts (you can even set `--test-policy=never` to keep it cheap if the goal is just protocol).
> > 2) Use `repo-agent sft-extract` to turn traces into step-level JSONL.
> > 3) Finetune the small model on that SFT to reduce `parse_errors` and get reliable tool calls.
>
> If you still don’t hit enough volume after extraction, it’s usually because of filtering (e.g., requiring tests pass). For “protocol compliance SFT”, you can loosen filters: keep steps where the tool call is valid even if the rollout eventually fails.
</question>


> 2) **Policy quality given valid tools** (choosing the next action + deciding when to patch). This is where loops / “keeps reading” vs “write_file now” lives.
>
> Patch correctness (3) is a third axis that may be out-of-scope for a first DPO project unless you’re willing to pay for lots of test-running rollouts.

So we want a smaller (7B or 14B, Qwen2.5-7B preferrably) LLM model to have:
tool-call discipline + reasoning format + loop avoidance + evidence-based edits.

> Note (Codex): “reasoning format” is optional for the model output here, because under `--tool-protocol=json` the assistant output is a JSON object. The “reasoning” you care about is really: **does it choose the right tool + args next**. You can still optionally include a short `thought` field in tool args (the agent supports that).

Hypothesis:
And we believe preference tuning to be best to solve this problem?

> Note (Codex): for *pure syntax / instruction-following*, SFT is usually the fastest win. For *behavior shaping* (avoid loops, reduce wasted calls, “patch sooner”), DPO makes more sense.
> A very reasonable scoped project is: **SFT to teach the JSON tool protocol → DPO to improve behavior**. You still “did DPO”, but you don’t ask DPO to do the job of SFT.

Preference tuning (DPO) a 7B model on preference pairs that come from an intelligent *teacher* will improve the metrics below like pass/fail rate, loops detected, num_steps. Also it should decrease the number of incorrect tool calls

> Note (Codex): I’d commit to the claim that DPO improves **tool-call validity + loop/efficiency metrics**. Claiming it improves pass rate is fine as a *hypothesis*, but in practice pass rate may not move much unless the model is already “close enough” on patching.

```json
"results": [
    {
      "task_id": "fix_quicksort",
      "run_id": "195896c323",
      "success": false,
      "steps": 0,
      "tool_calls": 0,
      "files_touched": [],
      "error": "Unknown LLM provider: 'together-chat'",
      "duration_s": 0.15348029136657715,
      "final_summary": "",
      "test_output": "",
      "metadata": {
        "difficulty": "easy",
        "category": "sorting",
        "expected_files": [
          "python_programs/quicksort.py"
        ]
      },
      "reflection_count": 0,
      "loop_detections": 0,
      "parse_errors": 0,
      "test_runs": 0,
      "tool_breakdown": {}
    }
```

> Note (Codex): that specific snippet is a driver config error (`Unknown LLM provider: 'together-chat'`), not a model failure. For baseline metrics you want runs that complete and produce:
> - `parse_errors` (invalid JSON tool actions)
> - `tool_breakdown` (does it ever call `write_file`?)
> - `loop_detections` / `steps`
> - `success` (optional; depends on whether you want to pay for tests)

One issue I see already is, when we run a 7B ootb model right now, it can't follow instructions at all so we don't even know what the metrics are. I don't want to overcomplicate it though, if it goes from not being able to follow instructions to being able to fix bugs, that's a huge signal. We can figure out ablations after the fact.

Data:
**Pairs:** cheap to generate because you can create them without a super-strong model:

* “Good” = follows protocol, uses tools in the right order, stops when stuck, produces minimal diff, cites failing test output, doesn’t thrash.
* “Bad” = violates invariants (edits unrelated files, ignores test failures, repeats loops, changes too much, doesn’t run tests, hallucinated file paths).

> Note (Codex): make “good vs bad” *operational* so you can label pairs automatically (no judge model).
> A simple, first-pass scoring rubric for a **single next-action**:
> - **Hard constraints (must-pass)**: valid JSON; `type ∈ {tool_call, final}`; if `tool_call` then `name ∈ {list_files, read_file, grep, write_file}`; args have required keys.
> - **Loop penalty**: exact repeat of the last tool call args; repeating `read_file` on the same file with tiny `max_chars`; repeated `grep` with same pattern+dir.
> - **Progress reward**: after a grep hit like `file.py:133:...`, prefer reading the relevant file with enough context (larger `max_chars`), and eventually prefer `write_file` once failure is localized.
>
> This is “behavior DPO”: you’re teaching *what to do next*, not proving the final patch is correct.

Example of a preference pair for the second iteration of the driver.
```json
{
  "input": {
    "messages": [
      { "role": "system", "content": "You are a repo-fixing agent. ... (same system prompt used at runtime)" },
      { "role": "user", "content": "GOAL:\\nFix the failing tests in this repo." },
      { "role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"grep\",\"args\":{\"pattern\":\"open(\",\"rel_dir\":\"pysnooper\",\"max_hits\":10}}" },
      { "role": "user", "content": "[tool_result]\\npysnooper/tracer.py:133:with open(self.path, 'w' if self.overwrite else 'a') as output_file:" }
    ]
  },
  "preferred_output": [
    { "role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{\"rel_path\":\"pysnooper/tracer.py\",\"max_chars\":4000}}" }
  ],
  "non_preferred_output": [
    { "role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{\"rel_path\":\"pysnooper/tracer.py\",\"max_chars\":200}}" }
  ]
}
```

> Note (Codex): where do `preferred_output` / `non_preferred_output` come from?
> - **Preferred**: a strong teacher’s next action at that state (e.g., `gpt-4.1-mini` or a strong Together model), captured from traces.
> - **Non-preferred**: the small model’s next action on the *same* state (or a synthetic corruption of the teacher action).
>
> This is the most cost-controlled way to get thousands of clean pairs: you only need **one-step completions**, not full rollouts with tests.

Contstructing the training data:
So if we have a multi-turn chat after 10 iterations, how many training samples do we think we can get out of that? It will depend, right?
So let's say we have 8 rollouts for a task, maybe 4 PASS and 4 FAIL. Are we only gonna look at the PASS cases? I mean, if we're trying to improve tool following instructions and reasoning/patch ability, hmmm....now I'm thinking the scope of this is too much, and maybe I should just do SFT to improve instruction following. but I want to do DPO. This where i could using your help in grounding the direction.

> Answer (Codex): if you do **step-level DPO**, each iteration can produce ~1 preference pair.
> - Rough math: `num_pairs ≈ tasks × rollouts × avg_steps_kept`.
> - QuixBugs Python has ~50 tasks. If you do 8 rollouts × ~12 kept steps, that’s ~4800 candidate pairs before filtering.
>
> **Do we only use PASS rollouts?**
> - For *behavior DPO* (tool discipline / loop avoidance): **no**, you can use FAIL rollouts too, as long as the *preferred* action is still “better behavior” at that state (valid JSON, non-looping, evidence-seeking, etc.).
> - For *patch-quality DPO* (functional correctness): **yes-ish** — you usually want labels tied to test outcomes, which is expensive.
>
> **Grounded scope recommendation (minimal but resume-worthy):**
> 1) Define the goal as: “Make a small model a reliable JSON-tool-using repo agent.”
> 2) Build DPO pairs where **preferred = teacher’s next JSON tool action**, **non-preferred = student’s next JSON tool action** for the *same input state*.
>    - This avoids needing pass/fail labels for every step.
>    - It directly trains the behavior you care about (tool calls + decision policy).
> 3) Evaluate on held-out QuixBugs tasks (e.g., 80/20 split) using metrics like: `parse_errors`, `loop_detections`, `avg_steps`, `% runs that reach a write_file`, and (optional) `success`.
>
> If you want a single “patching” hook without going broke: add a small subset of steps where the teacher writes a patch and tests pass, then preference-train “write_file now” vs “keep searching”.

> Concrete “project spec” (Codex):
> - **Inputs**: repo-agent runtime `system` prompt + `user goal` + tool results so far (exactly as in traces).
> - **Outputs (trained)**: one JSON object in assistant `content` (`{"type":"tool_call",...}` or `{"type":"final",...}`).
> - **Train data**: step-level DPO pairs (`preferred_output` teacher action, `non_preferred_output` student action) + optional small SFT warm-start for syntax.
> - **Primary success metric**: tool-call validity + fewer loops / fewer wasted steps on held-out tasks (pass rate is a “nice to have”).
> - **Deliverable**: (1) data generator script, (2) Together DPO finetune job, (3) before/after eval report on held-out suite.
