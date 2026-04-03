#!/usr/bin/env python3

import argparse
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Gemma PT model through Hugging Face Transformers."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id to load (default: {DEFAULT_MODEL_ID})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception as exc:  # pragma: no cover - runtime setup dependent
        print(f"Failed to load {args.model_id}: {exc}", file=sys.stderr)
        print(
            "If this is a gated Gemma repository, accept the license on Hugging Face "
            "and make sure you are logged in with huggingface-cli login.",
            file=sys.stderr,
        )
        return 1

    param = next(model.parameters(), None)
    device = str(param.device) if param is not None else "unknown"

    print("Gemma PT loaded successfully")
    print(f"Model ID: {args.model_id}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"First parameter device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
