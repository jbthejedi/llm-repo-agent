In this document are the contents from a chat I had with gpt5.2 on how to dpo finetune Qwen using Together AI's python client

Yes—you can kick off a Together fine-tune job from a Python CLI, and DPO + LoRA is supported via the `training_method` + `training_type` fields on `/v1/fine-tunes`. ([GitHub][1])

One important constraint: **the “Turbo” variants typically aren’t the finetunable base models**. Together’s fine-tuning model list includes things like `Qwen/Qwen2.5-7B-Instruct` (and other “Reference” models), not “*-Turbo”. So in practice you’ll likely fine-tune `Qwen/Qwen2.5-7B-Instruct` (or another listed finetune base) rather than `Qwen-7B-Instruct-Turbo`. ([docs.together.ai][2])

## Dataset format (preference pairs)

Your sample shape matches Together’s preference fine-tuning format: each example contains `input` and a preferred vs non-preferred output. ([docs.together.ai][3])
In the actual training file, store it as **JSONL** (one example per line).

---

## CLI script: `poetry run python dpo.py --dataset data.jsonl ...`

```python
# dpo.py
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

from together import Together


def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".jsonlines"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e


def _ensure_jsonl(dataset_path: Path) -> Tuple[Path, bool]:
    """
    Returns (jsonl_path, is_temp).
    Accepts:
      - .jsonl (already ok)
      - .json containing a list of examples
    """
    if _is_jsonl(dataset_path):
        return dataset_path, False

    if dataset_path.suffix.lower() != ".json":
        raise ValueError("Dataset must be .jsonl or .json (list of examples).")

    data = _load_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError("If dataset is .json, it must be a JSON array (list) of examples.")

    tmp = Path(tempfile.mkstemp(prefix="together_dpo_", suffix=".jsonl")[1])
    with tmp.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return tmp, True


def _validate_example(ex: Dict[str, Any]) -> None:
    # Minimal sanity checks for Together preference format
    for k in ("input", "preferred_output", "non_preferred_output"):
        if k not in ex:
            raise ValueError(f"Missing key '{k}' in example: {list(ex.keys())}")

    if not isinstance(ex["input"], dict):
        raise ValueError("'input' must be an object.")
    if "messages" not in ex["input"] or not isinstance(ex["input"]["messages"], list):
        raise ValueError("'input.messages' must be a list of chat messages.")

    if not isinstance(ex["preferred_output"], list) or not ex["preferred_output"]:
        raise ValueError("'preferred_output' must be a non-empty list of messages.")
    if not isinstance(ex["non_preferred_output"], list) or not ex["non_preferred_output"]:
        raise ValueError("'non_preferred_output' must be a non-empty list of messages.")


def main() -> None:
    p = argparse.ArgumentParser(description="Kick off Together DPO+LoRA fine-tuning.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to .jsonl or .json (list) dataset.")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base finetune model name (must be supported by Together fine-tuning).",
    )
    p.add_argument("--suffix", type=str, default="dpo-lora", help="Suffix for the output model name.")

    # DPO knobs (Together OpenAPI fields)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--dpo-reference-free", action="store_true")
    p.add_argument("--dpo-normalize-by-length", action="store_true")
    p.add_argument("--rpo-alpha", type=float, default=0.0, help="Optional; leave 0 unless you mean to use it.")
    p.add_argument("--simpo-gamma", type=float, default=0.0, help="Optional; leave 0 unless you mean to use it.")

    # LoRA knobs (Together OpenAPI fields)
    p.add_argument("--lora-r", type=int, default=16, required=False)
    p.add_argument("--lora-alpha", type=int, default=32, required=False)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-trainable-modules", type=str, default="all-linear")

    # General training knobs
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=str, default="max", help='Integer or "max".')
    p.add_argument("--learning-rate", type=float, default=1e-5)

    # Optional: poll until done
    p.add_argument("--watch", action="store_true", help="Poll job status until terminal.")
    p.add_argument("--poll-seconds", type=int, default=10)

    args = p.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("Set TOGETHER_API_KEY in your environment.")

    dataset_path, is_temp = _ensure_jsonl(args.dataset)

    # Validate a few rows (fast fail)
    it = _iter_jsonl(dataset_path)
    for _ in range(3):
        try:
            ex = next(it)
        except StopIteration:
            break
        _validate_example(ex)

    client = Together(api_key=api_key)

    # 1) Upload dataset file
    # Together docs show using `client.files.upload(...)` for fine-tuning datasets.
    uploaded = client.files.upload(file=str(dataset_path))
    training_file_id = uploaded.id

    # 2) Create DPO + LoRA fine-tune
    job = client.fine_tuning.create(
        model=args.model,
        training_file=training_file_id,
        suffix=args.suffix,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_method={
            "method": "dpo",
            "dpo_beta": args.dpo_beta,
            "dpo_reference_free": bool(args.dpo_reference_free),
            "dpo_normalize_logratios_by_length": bool(args.dpo_normalize_by_length),
            "rpo_alpha": args.rpo_alpha,
            "simpo_gamma": args.simpo_gamma,
        },
        training_type={
            "type": "Lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_trainable_modules": args.lora_trainable_modules,
        },
    )

    print(json.dumps(
        {
            "job_id": job.id,
            "status": getattr(job, "status", None),
            "model_output_name": getattr(job, "model_output_name", None),
            "training_file": training_file_id,
        },
        indent=2
    ))

    if args.watch:
        import time

        terminal = {"completed", "failed", "cancelled"}
        while True:
            cur = client.fine_tuning.retrieve(id=job.id)
            status = str(cur.status)
            print(json.dumps(
                {
                    "job_id": cur.id,
                    "status": status,
                    "model_output_name": getattr(cur, "model_output_name", None),
                    "updated_at": getattr(cur, "updated_at", None),
                },
                indent=2
            ))
            if status.lower() in terminal:
                break
            time.sleep(args.poll_seconds)

    if is_temp:
        try:
            dataset_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
```

### Notes on what this is using (from Together docs / spec)

* Uploading training data with the Python SDK via `client.files.upload(...)`. ([docs.together.ai][4])
* Creating a fine-tune job via `client.fine_tuning.create(...)`. ([docs.together.ai][5])
* DPO is selected by `training_method: {"method": "dpo", ...}` and LoRA by `training_type: {"type": "Lora", ...}`. ([GitHub][1])
* Preference pair example format (`input` + `preferred_output` + `non_preferred_output`). ([docs.together.ai][3])

If you paste your exact schema for a single example (one line from your JSONL), I can sanity-check it against the preference format Together expects without you having to run a job.

[1]: https://raw.githubusercontent.com/togethercomputer/openapi/main/openapi.yaml "raw.githubusercontent.com"
[2]: https://docs.together.ai/docs/fine-tuning-models "Supported Models - Together.ai Docs"
[3]: https://docs.together.ai/docs/preference-fine-tuning "Preference Fine-Tuning - Together.ai Docs"
[4]: https://docs.together.ai/reference/upload-file "Upload a file - Together.ai Docs"
[5]: https://docs.together.ai/docs/fine-tuning-python "Python Library - Together.ai Docs"
