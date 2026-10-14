import argparse
import json
import os
import time

from llm import call_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local Hugging Face generation.")
    parser.add_argument("--model", default=os.environ.get("LOCAL_MODEL_ID"), required=False)
    parser.add_argument(
        "--prompt",
        default="Write a Python function named compose that returns the integer 1.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "low", "medium", "high"],
        help="reasoning level for this call (default: LOCAL_REASONING / off)",
    )
    args = parser.parse_args()
    if args.reasoning:
        os.environ["LOCAL_REASONING"] = args.reasoning

    if not args.model:
        parser.error("provide --model or set LOCAL_MODEL_ID")

    started = time.perf_counter()
    content, usage = call_llm(args.prompt, args.model, "")
    result = {
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": usage,
        "content": content,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()