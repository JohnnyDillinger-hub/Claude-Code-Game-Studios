#!/usr/bin/env python3

import argparse
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single inference call against a Gemma PT model."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id to load (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text to run. Mutually exclusive with --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a file containing the prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling cutoff.",
    )
    args = parser.parse_args()

    if bool(args.prompt) == bool(args.prompt_file):
        parser.error("Provide exactly one of --prompt or --prompt-file.")

    return args


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt

    with open(args.prompt_file, "r", encoding="utf-8") as handle:
        return handle.read()


def main() -> int:
    args = parse_args()
    prompt = load_prompt(args)

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

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
