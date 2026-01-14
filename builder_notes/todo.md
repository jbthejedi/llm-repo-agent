- test repo agent (select llm model from together) on harder repo (Defects4J, ManyBugs, Bugs.jar, BugsInPy). -> BugsInPy first
- create sft dataset with more tasks (Defects4J, ManyBugs, Bugs.jar)
- verify DeepSeek-R1-Distilled-Qwen-14B works sft/dpo/endpoint
- after (after?) sft of model, estimate cost for generating dpo
- I ran the `sft` command using `--provider openai` and `--model gpt-4.1-mini` and `--tool-protocol json` and all the tests failed, but it worked with `qwen2.5-72b-instruct-turbo`. Need to look into that

