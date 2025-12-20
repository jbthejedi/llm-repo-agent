## Appendix

<!-- <a id="trace-output">
**Figure 3** -->

### Result prints
```bash
(base) ➜  llm-repo-agent git:(trace-run-metadata) ✗ poetry run repo-agent \                                
  --repo ~/projects/QuixBugs \
  --goal "Fix quicksort so python_testcases/test_quicksort.py passes. Make the smallest correct change." \
  --trace runs/quixbugs_trace.jsonl \
  --test "python -m pytest -q python_testcases/test_quicksort.py"
Fixed quicksort by changing the partition conditions to use '<' for lesser and '>=' for greater to correctly sort duplicates. All tests pass now.
Tests: PASSED - All tests passed.
Output snippet: .............                                                            [100%]
```


### Full prompt example
```bash
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run bac3f20feb --kind llm_request --index 0 --prompt-with-history --preserve-newlines --history-window 2
PROMPT:
  system: You are a repo-fixing agent.
  You operate in a loop: choose ONE action, then wait for the tool result.
  
  OUTPUT CONTRACT (STRICT):
  - Output EXACTLY ONE JSON object. No extra text. No markdown.
  - Never output multiple JSON objects.
  - If your response accidentally contains multiple JSON objects or trailing text, the agent will parse only 
the first JSON object and ignore the rest.                                                                     - If more work is needed, choose the single best next tool_call and stop.
  
  ALLOWED ACTIONS:
  A) {"type":"tool_call","name":<tool_name>,"args":{...}}
  B) {"type":"final","summary":"...","changes":[{"path":"...","description":"..."}]}
  
  EXAMPLES:
  Example tool_call: {"type":"tool_call","name":"list_files","args":{"rel_dir":".","max_files":20}}
  Example final: {"type":"final","summary":"Found test command: pytest","changes":[...
  
  user: GOAL:
  Fix quicksort so python_testcases/test_quicksort.py passes. Make the smallest correct change.
  
  STATE (compact):
  {
    "state": {
      "notes": [],
      "files_touched": [],
      "last_test": null,
      "run_id": null
    },
    "history": []
  }

HISTORY (events 0..3 around selected index 1):
 - 0: 2025-12-18 12:19:56 [run_start] {"run_id": "bac3f20feb", "goal": "Fix quicksort so python_testcases/tes
t_quicksort.py passes. Make the smallest correct change."}                                                    - 1: 2025-12-18 12:19:56 [llm_request] system: You are a repo-fixing agent. You operate in a loop: choose ON
E action, then wait for the tool result.  OUTPUT CONTRACT (STRICT): - Output EXACTLY ONE JSON object. No extra text. No markdown. - Never output multiple JSON objects. - If your response accidentally contains multiple JSON objects ...                                                                                              - 2: 2025-12-18 12:19:57 [llm_action] {"t": 0, "raw": {"type": "tool_call", "name": "list_files", "args": {"
rel_dir": "python_testcases", "max_files": 20}}, "action": {"type": "tool_call", "name": "list_files", "args": {"rel_dir": "python_testcases", "max_files": 20}}}                                                          - 3: 2025-12-18 12:19:57 [tool_result] tool_result {"t": 0, "tool": "list_files", "args": {"rel_dir": "pytho
n_testcases", "max_files": 20}, "obs": {"ok": true, "output": "python_testcases/test_lcs_length.py\npython_testcases/test_breadth_first_search.py\npython_testcases/test_pascal.py\npython_testcases/test_sqrt.py\npython_testcases/tes...                                                                                             
```

### Example: how RunSummary works to inform the LLM
#### From the same run

t == 6
Before LLM called.
So we can see that a tool_call::write_file was executed by the Controller, and quicksort was amended.

```python
{'notes': ['touched python_programs/quicksort.py'],
 'files_touched': ['python_programs/quicksort.py'],
 'last_test': {'ok': True,
               'output': '.............'
                         '[100%]\n'
                         '13 passed in 0.02s'},
 'run_id': '7f1d4b42cd'}
```

After LLM called

```python
pprint.pp(parsed_action.to_dict())
{'type': 'final',
 'summary': 'Fixed quicksort to include elements equal to the pivot in the '
            "'greater' partition by changing the comparison from '>' to '>=' "
            'in python_programs/quicksort.py. All tests in '
            'python_testcases/test_quicksort.py pass now.',
 'changes': [{'path': 'python_programs/quicksort.py',
              'description': "Changed comparison in quicksort from '>' to '>=' "
                             'for the greater partition to correctly handle '
                             'elements equal to the pivot.'}]}
```

Then

```bash
Fixed quicksort to include elements equal to the pivot in the 'greater' partition by changing the comparison from '>' to '>=' in python_programs/quicksort.py. All tests in python_testcases/test_quicksort.py pass now.
Tests: PASSED - All tests passed.
Output snippet: .............                                                            [100%]
```