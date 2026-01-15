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

    tmp = Path(tempfile.mkstemp(prefix="together_sft_", suffix=".jsonl")[1])
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


def _parse_train_on_inputs(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    if normalized == "auto":
        return "auto"
    raise ValueError("train-on-inputs must be one of: auto, true, false.")


def _validate_message(msg: Dict[str, Any]) -> None:
    if not isinstance(msg, dict):
        raise ValueError("Each message must be an object.")
    role = msg.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"Unsupported role: {role!r}")
    if "content" not in msg and "tool_calls" not in msg:
        raise ValueError("Message must include 'content' or 'tool_calls'.")
    if role == "tool" and "content" not in msg:
        raise ValueError("Tool messages must include 'content'.")


def _validate_example(ex: Dict[str, Any]) -> None:
    if "messages" not in ex:
        raise ValueError("Missing key 'messages' in example.")
    if not isinstance(ex["messages"], list) or not ex["messages"]:
        raise ValueError("'messages' must be a non-empty list of chat messages.")
    for msg in ex["messages"]:
        _validate_message(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick Together SFT+LoRA fine-tune using a chat dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("runs/instruction_tuning_test_3/sft_dataset.jsonl"),
        help="Path to .jsonl (or .json list) SFT dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-Turbo",
        help="Base fine-tune model name (must be supported by Together fine-tuning).",
    )
    parser.add_argument("--suffix", type=str, default="sft-lora", help="Suffix for the output model name.")

    # General training knobs
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=str, default="max", help='Integer or "max".')
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Warmup ratio (fraction of training steps), e.g. 0.05. If omitted, provider default is used.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Max gradient norm for clipping, e.g. 1.0. If omitted, provider default is used.",
    )
    parser.add_argument(
        "--train-on-inputs",
        type=str,
        default=None,
        help="Whether to compute loss on user/input tokens: auto|true|false (if omitted, provider default is used).",
    )
    parser.add_argument(
        "--learning-rate-scheduler-type",
        dest="learning_rate_scheduler_type",
        type=str,
        default=None,
        help="Learning rate scheduler type (provider-dependent), e.g. linear|cosine (if omitted, provider default is used).",
    )

    # LoRA knobs
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable LoRA (default: enabled).")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-trainable-modules", type=str, default="all-linear")

    # Optional: poll until done
    parser.add_argument("--watch", action="store_true", help="Poll job status until terminal.")
    parser.add_argument("--poll-seconds", type=int, default=10)
    # Optional: Weights & Biases logging
    parser.add_argument("--wandb-api-key", type=str, default=None,
                        help="Weights & Biases API key (defaults to WANDB_API_KEY env var).")
    parser.add_argument("--wandb-project-name", type=str, default=None,
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Weights & Biases run name.")

    args = parser.parse_args()

    load_dotenv()
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    if not together_api_key:
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

    client = Together(api_key=together_api_key)
    uploaded = client.files.upload(file=str(dataset_path))
    training_file_id = uploaded.id

    batch_size = _parse_batch_size(args.batch_size)

    job_kwargs = dict(
        model=args.model,
        training_file=training_file_id,
        suffix=args.suffix,
        n_epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        training_method="sft",
    )
    if args.warmup_ratio is not None:
        if not (0.0 <= args.warmup_ratio <= 1.0):
            raise ValueError("--warmup-ratio must be between 0.0 and 1.0")
        job_kwargs["warmup_ratio"] = args.warmup_ratio
    if args.max_grad_norm is not None:
        if args.max_grad_norm <= 0:
            raise ValueError("--max-grad-norm must be positive")
        job_kwargs["max_grad_norm"] = args.max_grad_norm
    if args.train_on_inputs is not None:
        job_kwargs["train_on_inputs"] = _parse_train_on_inputs(args.train_on_inputs)
    if args.learning_rate_scheduler_type is not None:
        job_kwargs["learning_rate_scheduler_type"] = args.learning_rate_scheduler_type
    wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        job_kwargs["wandb_api_key"] = wandb_api_key
    if args.wandb_project_name:
        job_kwargs["wandb_project_name"] = args.wandb_project_name
    if args.wandb_name:
        job_kwargs["wandb_name"] = args.wandb_name
    if args.lora:
        job_kwargs.update(
            lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_trainable_modules=args.lora_trainable_modules,
        )

    job = client.fine_tuning.create(**job_kwargs)

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

        terminal = {
            "completed",
            "completed_with_warnings",
            "failed",
            "cancelled",
            "canceled",
            "succeeded",
            "succeeded_with_warnings",
            "success",
            "errored",
            "error",
        }
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
