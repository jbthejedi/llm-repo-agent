The new source code and a summary of all the changes I suggested to codex

Prompts: Added a Prompt class to assemble system/user messages instead of ad-hoc dicts in RepoAgent.run—cleaner construction and easier to extend.

State → Summary: Replaced the ad-hoc State with a derived RunSummary + summarize_history (in summary.py, with state.py as a shim). Reasoning: make History the source of truth and pass a compact digest to the model, reducing noise and implicit mutable state.

History as the ledger: Added DriverNoteEvent and append_driver_note, plus touched_files() in History. write_file now emits “touched <path>” notes; loop/trailing-text warnings are logged as driver notes. Reasoning: side effects and driver hints belong in the transcript, not in a side scratchpad.

Agent loop wiring: RepoAgent now computes a summary from History each turn and feeds that to the prompt; controller no longer takes State. Final test reporting uses the derived summary. Reasoning: derive facts from the ledger, avoid out-of-band mutation, keep prompts concise with a curated summary.

Tests: Renamed/updated test_summary_history.py and added coverage for summarize_history; silenced pytest trying to collect the TestResult dataclass. Reasoning: keep tests aligned with the new summary model and avoid noisy warnings.

Why it’s better: clearer ownership (History = evidence; RunSummary = digest), less hidden mutable state, cleaner prompt inputs, and explicit logging of driver signals and touched files for auditability.