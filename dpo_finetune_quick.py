from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from dotenv import load_dotenv


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


def _parse_batch_size(value: str) -> int | str:
    if value == "max":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("batch-size must be an integer or 'max'.") from exc
    if parsed <= 0:
        raise ValueError("batch-size must be positive.")
    return parsed


def _validate_example(ex: Dict[str, Any]) -> None:
    for key in ("input", "preferred_output", "non_preferred_output"):
        if key not in ex:
            raise ValueError(f"Missing key '{key}' in example: {list(ex.keys())}")

    if not isinstance(ex["input"], dict):
        raise ValueError("'input' must be an object.")
    if "messages" not in ex["input"] or not isinstance(ex["input"]["messages"], list):
        raise ValueError("'input.messages' must be a list of chat messages.")

    if not isinstance(ex["preferred_output"], list) or not ex["preferred_output"]:
        raise ValueError("'preferred_output' must be a non-empty list of messages.")
    if not isinstance(ex["non_preferred_output"], list) or not ex["non_preferred_output"]:
        raise ValueError("'non_preferred_output' must be a non-empty list of messages.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick Together DPO+LoRA fine-tune using a preference dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("runs/prefs/dpo_dataset_pilot.jsonl"),
        help="Path to .jsonl (or .json list) preference dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base fine-tune model name (must be supported by Together fine-tuning).",
    )
    parser.add_argument("--suffix", type=str, default="dpo-lora", help="Suffix for the output model name.")

    # DPO knobs
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-normalize-by-length", action="store_true")
    parser.add_argument("--rpo-alpha", type=float, default=0.0)
    parser.add_argument("--simpo-gamma", type=float, default=0.0)

    # LoRA knobs
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-trainable-modules", type=str, default="all-linear")

    # General training knobs
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=str, default="max", help='Integer or "max".')
    parser.add_argument("--learning-rate", type=float, default=1e-5)

    # Optional: poll until done
    parser.add_argument("--watch", action="store_true", help="Poll job status until terminal.")
    parser.add_argument("--poll-seconds", type=int, default=10)

    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("Set TOGETHER_API_KEY in your environment.")

    try:
        from together import Together
    except Exception as exc:
        raise RuntimeError("Missing Together SDK. Install with `pip install together`.") from exc

    dataset_path, is_temp = _ensure_jsonl(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    it = _iter_jsonl(dataset_path)
    for _ in range(3):
        try:
            ex = next(it)
        except StopIteration:
            break
        _validate_example(ex)

    client = Together(api_key=api_key)

    uploaded = client.files.upload(file=str(dataset_path))
    training_file_id = uploaded.id

    batch_size = _parse_batch_size(args.batch_size)

    job = client.fine_tuning.create(
        model=args.model,
        training_file=training_file_id,
        suffix=args.suffix,
        n_epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        training_method="dpo",
        dpo_beta=args.dpo_beta,
        dpo_normalize_logratios_by_length=bool(args.dpo_normalize_by_length),
        rpo_alpha=args.rpo_alpha,
        simpo_gamma=args.simpo_gamma,
        lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_trainable_modules=args.lora_trainable_modules,
    )

    print(json.dumps(
        {
            "job_id": job.id,
            "status": getattr(job, "status", None),
            "model_output_name": getattr(job, "model_output_name", None),
            "training_file": training_file_id,
        },
        indent=2,
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
                indent=2,
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
