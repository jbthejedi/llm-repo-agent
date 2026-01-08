# Run eval suite using Together Base Qwen model
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/my_eval_6 \
    --report runs/my_eval_6/report.json \
    --llm-provider together \
    --model Qwen/Qwen2.5-72B-Instruct-Turbo 

# Run eval suite using DPO finetuned model
# endpoint_id: justinbarrye_c241/Qwen2.5-7B-Instruct-dpo-lora-4bf2fea2-4d8be882
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/my_eval_6 \
    --report runs/my_eval_6/report.json \
    --llm-provider together \
    --model justinbarrye_c241/Qwen2.5-7B-Instruct-dpo-lora-4bf2fea2-4d8be882

# DPO finetune
 poetry run repo-agent prefs \
  --suite eval/suites/pref_data_gen_pilot_1.json \
  --rollouts 4 \            
  --out runs/prefs/dpo_dataset_pilot.jsonl \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.7 \
  --seed 42