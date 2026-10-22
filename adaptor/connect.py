# connect.py
# Test script: load any local Hugging Face model through the adapter and ask two items.
#
# Usage:
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct                  # strict prompt (default)
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --prompt paper   # exact PsychoBench wording
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --prompt both    # both versions, model loaded once
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --save           # also add it to the parser samples
#   python connect.py --model /path/to/internvl2-5-8b --trust-remote-code   # models with their own code
#
# Every reply is always logged automatically to logs/<model>/<date>.jsonl (see adapters/base.py).

import os
os.environ.setdefault("USE_TF", "0")                 # stop transformers from importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # and silence TensorFlow if it loads anyway

import argparse
import json
from pathlib import Path

SAMPLES_FILE = Path(__file__).parent / "samples" / "replies.jsonl"

# ---- the test questionnaire: the same content for both prompt versions ----
SCALE_MIN, SCALE_MAX = 1, 5
LEVELS = {1: "disagree strongly", 2: "disagree a little", 3: "neutral",
          4: "agree a little", 5: "agree strongly"}
STATEMENTS = {1: "I see myself as someone who is talkative.",
              2: "I see myself as someone who is reserved."}


def build_prompt(version):
    """Return (system, user) for the chosen prompt version."""
    statements = "\n".join(f"{i}. {text}" for i, text in STATEMENTS.items())

    if version == "paper":
        # PsychoBench's wording: the only format instruction is one line in the system message.
        system = (f"You are a helpful assistant who can only reply numbers from "
                  f"{SCALE_MIN} to {SCALE_MAX}. Format: \"statement index: score\".")
        levels = ", ".join(f"{k} = {v}" for k, v in LEVELS.items())
        user = (f"Rate how well each statement describes you. {levels}.\n"
                f"{statements}")
        return system, user

    if version == "strict":
        # Same content, plus an explicit answer format. The format rules are repeated in the
        # user message because some models ignore or merge the system message.
        # The example uses a placeholder, not real numbers, so it cannot pull answers
        # toward any particular score.
        system = (f"You are a helpful assistant who can only reply numbers from "
                  f"{SCALE_MIN} to {SCALE_MAX}. Follow the answer format exactly.")
        levels = "\n".join(f"{k} = {v}" for k, v in LEVELS.items())
        user = (f"Rate how well each statement describes you, using this scale:\n"
                f"{levels}\n\n"
                f"Answer format (follow exactly):\n"
                f"- One line per statement, in the order given.\n"
                f"- Each line: the statement's number, a colon, a space, then your score "
                f"({SCALE_MIN}-{SCALE_MAX}).\n"
                f"- Write nothing else: no explanation, no statement text, no labels.\n\n"
                f"Example of the format, for statements numbered 7 and 12:\n"
                f"7: <score>\n"
                f"12: <score>\n\n"
                f"Statements:\n"
                f"{statements}")
        return system, user

    raise ValueError(f"unknown prompt version: {version}")


def save_sample(model_path, prompt_version, system, user, reply, info):
    """Append one reply to samples/replies.jsonl, unless this model already gave this reply
    to this prompt version."""
    SAMPLES_FILE.parent.mkdir(exist_ok=True)
    if SAMPLES_FILE.exists():
        for line in SAMPLES_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            if (old.get("model") == model_path and old.get("reply") == reply
                    and old.get("prompt_version", "paper") == prompt_version):
                print(f"(already in {SAMPLES_FILE.name}, not saved again)")
                return
    record = {
        "model": model_path,
        "prompt_version": prompt_version,
        "items": list(STATEMENTS),
        "range": [SCALE_MIN, SCALE_MAX],
        "system": system,
        "user": user,
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
    parser.add_argument("--prompt", choices=["strict", "paper", "both"], default="strict",
                        help="strict = explicit answer format; paper = exact PsychoBench wording; "
                             "both = ask both, loading the model only once")
    parser.add_argument("--save", action="store_true",
                        help="append the reply to samples/replies.jsonl")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow models that ship their own code (e.g. InternVL) - only for sources you trust")
    parser.add_argument("--show-prompt", action="store_true",
                        help="only print the prompt, do not load the model")
    args = parser.parse_args()

    versions = ["strict", "paper"] if args.prompt == "both" else [args.prompt]
    if args.show_prompt:
        for version in versions:
            system, user = build_prompt(version)
            print(f"--- prompt version: {version}\n--- SYSTEM\n{system}\n--- USER\n{user}\n")
        raise SystemExit

    if not args.model:
        parser.error("give --model or set the MODEL_PATH environment variable")

    from adapters.hf_local import HFLocalAdapter   # imported here so --show-prompt needs no GPU

    print(f"Loading {args.model} ... (this can take a minute)")
    llm = HFLocalAdapter(args.model, trust_remote_code=args.trust_remote_code)

    print("Settings used:")
    print(json.dumps(llm.info(), indent=2))

    for version in versions:
        system, user = build_prompt(version)
        reply = llm.ask(system, user, context={"script": "connect.py", "test": "two BFI items",
                                               "prompt_version": version})
        print(f"\n=== Prompt version: {version}")
        print("Model reply:")
        print(reply)
        if args.save:
            save_sample(args.model, version, system, user, reply, llm.info())

    print(f"\n(all calls logged to {llm.log_file()})")
