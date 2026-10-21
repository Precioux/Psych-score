# connect.py
# Test script: load any local Hugging Face model through the adapter and ask two items.
#
# Usage:
#   python connect.py --model Qwen/Qwen2-VL-7B-Instruct
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct

import os
os.environ.setdefault("USE_TF", "0")                 # stop transformers from importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # and silence TensorFlow if it loads anyway

import argparse
import json

from adapters.hf_local import HFLocalAdapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="Hugging Face model name or local folder")
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

    print("Model reply:")
    print(llm.ask(system, user))
