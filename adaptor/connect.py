# connect.py
# Test script: load any local Hugging Face model through the adapter and ask two items.
#
# Usage:
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --save   # also add it to the parser samples
#
# Every reply is always logged automatically to logs/<model>/<date>.jsonl (see adapters/base.py).

import os
os.environ.setdefault("USE_TF", "0")                 # stop transformers from importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # and silence TensorFlow if it loads anyway

import argparse
import json
from pathlib import Path

from adapters.hf_local import HFLocalAdapter

SAMPLES_FILE = Path(__file__).parent / "samples" / "replies.jsonl"


def save_sample(model_path, items, score_range, reply, info):
    """Append one reply to samples/replies.jsonl, unless the same model gave the same reply before."""
    SAMPLES_FILE.parent.mkdir(exist_ok=True)
    if SAMPLES_FILE.exists():
        for line in SAMPLES_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            if old.get("model") == model_path and old.get("reply") == reply:
                print(f"(already in {SAMPLES_FILE.name}, not saved again)")
                return
    record = {
        "model": model_path,
        "items": items,
        "range": score_range,
        "reply": reply,
        "expected": None,   # fill in by hand: the scores a correct parser must extract
        "note": "",         # fill in by hand: what is unusual about this reply
        "info": info,
    }
    with open(SAMPLES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved to {SAMPLES_FILE} - open it and fill in \"expected\" and \"note\".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="Hugging Face model name or local folder")
    parser.add_argument("--save", action="store_true",
                        help="append the reply to samples/replies.jsonl")
    args = parser.parse_args()
    if not args.model:
        parser.error("give --model or set the MODEL_PATH environment variable")

    print(f"Loading {args.model} ... (this can take a minute)")
    llm = HFLocalAdapter(args.model)

    print("Settings used:")
    print(json.dumps(llm.info(), indent=2))

    system = ('You are a helpful assistant who can only reply numbers from 1 to 5. '
              'Format: "statement index: score".')
    user = ("Rate how well each statement describes you. "
            "1 = disagree strongly, 2 = disagree a little, 3 = neutral, "
            "4 = agree a little, 5 = agree strongly.\n"
            "1. I see myself as someone who is talkative.\n"
            "2. I see myself as someone who is reserved.")

    reply = llm.ask(system, user, context={"script": "connect.py", "test": "two BFI items"})
    print("Model reply:")
    print(reply)
    print(f"(logged to {llm.log_file()})")

    if args.save:
        save_sample(args.model, [1, 2], [1, 5], reply, llm.info())
