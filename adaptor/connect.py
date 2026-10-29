# connect.py
# Test script: load any local Hugging Face model through the adapter and ask two items.
#
# Usage:
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct                  # strict prompt (default)
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --prompt paper   # exact PsychoBench wording
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --prompt both    # both versions, model loaded once
#   python connect.py --model Qwen/Qwen3-VL-8B-Instruct --save           # also add it to the parser samples
#   python connect.py --model /path/to/internvl2-5-8b --trust-remote-code   # models with their own code
#   python connect.py --models-file models.txt          # test every model in the list, one experiment
#
# Every reply is always logged automatically to logs/<model>/<date>.jsonl (see adapters/base.py).

import os
os.environ.setdefault("USE_TF", "0")                 # stop transformers from importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # and silence TensorFlow if it loads anyway

import argparse
import datetime
import gc
import json
import re
import time
import traceback
from pathlib import Path

SAMPLES_FILE = Path(__file__).parent / "samples" / "replies.jsonl"
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

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




# ---- testing many models in one experiment ----

def read_models_file(path):
    """One model per line: <path or HF name> [trust_remote_code]. Lines starting with # are ignored."""
    models = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        models.append({"model": parts[0], "trust_remote_code": "trust_remote_code" in parts[1:]})
    return models


def check_reply(reply):
    """Two checks on a reply to the two test statements.
    strict_format: exactly two lines "1: n" and "2: n" with n in range, nothing else.
    items_found:   both items can be read, even if the reply has extra text
                   (item number at the start of a line, then one number in range on that line)."""
    lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
    strict = (len(lines) == len(STATEMENTS) and all(
        re.fullmatch(rf"{i}\s*:\s*[{SCALE_MIN}-{SCALE_MAX}]", l) for i, l in zip(STATEMENTS, lines)))
    found = {}
    for l in lines:
        m = re.match(r"^\W*(\d+)\s*[.:)]", l)
        if not m or int(m.group(1)) not in STATEMENTS:
            continue
        rest = l[m.end():]
        scores = [int(x) for x in re.findall(r"\b(\d+)\b", rest) if SCALE_MIN <= int(x) <= SCALE_MAX]
        if scores:
            found[int(m.group(1))] = scores[-1]
    return strict, found


def free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_experiment(models, versions, save):
    """Load each model in turn, ask the test, record what happened, free the GPU, continue."""
    from adapters.hf_local import HFLocalAdapter

    exp_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    for n, entry in enumerate(models, 1):
        model = entry["model"]
        print(f"\n########## [{n}/{len(models)}] {model} ##########")
        row = {"model": model, "trust_remote_code": entry["trust_remote_code"],
               "status": None, "load_seconds": None, "answers": {}, "error": None}
        if model.startswith("/") and not Path(model).exists():
            row["status"] = "missing"
            row["error"] = "folder not found"
            print("  -> folder not found, skipped")
            results.append(row)
            continue

        llm = None
        try:
            start = time.time()
            llm = HFLocalAdapter(model, trust_remote_code=entry["trust_remote_code"])
            row["load_seconds"] = round(time.time() - start, 1)
            row["info"] = llm.info()
            print(f"  loaded in {row['load_seconds']}s | loader {row['info']['loader']} | "
                  f"{row['info']['dtype']} | {row['info']['device']}")
            for version in versions:
                system, user = build_prompt(version)
                start = time.time()
                reply = llm.ask(system, user, context={"script": "connect.py", "experiment": exp_id,
                                                       "test": "two BFI items",
                                                       "prompt_version": version})
                strict, found = check_reply(reply)
                row["answers"][version] = {"reply": reply, "seconds": round(time.time() - start, 1),
                                           "strict_format": strict, "items_found": found}
                print(f"  [{version}] reply: {reply!r}")
                if save:
                    save_sample(model, version, system, user, reply, llm.info())
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
            print(f"  -> FAILED: {row['error'][:300]}")
        finally:
            del llm
            free_gpu_memory()
        results.append(row)

    # ---- summary ----
    print("\n" + "=" * 100)
    print(f"EXPERIMENT {exp_id}: {len(models)} models, prompt version(s): {', '.join(versions)}")
    print("=" * 100)
    print(f"{'MODEL':<32} {'STATUS':<8} {'LOAD s':>7}  {'PROMPT':<7} {'FORMAT':<7} {'ITEMS':<6} REPLY")
    for r in results:
        name = r["model"].rstrip("/").split("/")[-1][:32]
        load = r["load_seconds"] if r["load_seconds"] is not None else "-"
        if not r["answers"]:
            print(f"{name:<32} {r['status']:<8} {load:>7}  {'-':<7} {'-':<7} {'-':<6} {(r['error'] or '')[:60]}")
        for version, a in r["answers"].items():
            items = "both" if len(a["items_found"]) == len(STATEMENTS) else f"{len(a['items_found'])}/{len(STATEMENTS)}"
            first = a["reply"].replace("\n", " | ")[:60]
            print(f"{name:<32} {r['status']:<8} {load:>7}  {version:<7} "
                  f"{'ok' if a['strict_format'] else 'messy':<7} {items:<6} {first}")
    n_ok = sum(r["status"] == "ok" for r in results)
    n_fail = sum(r["status"] == "failed" for r in results)
    n_missing = sum(r["status"] == "missing" for r in results)
    print(f"\n{n_ok} answered, {n_fail} failed, {n_missing} missing.")
    print("FORMAT ok = exactly the requested format; ITEMS both = both scores could be read anyway.")

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / f"connect_test_{exp_id}.json"
    out.write_text(json.dumps({"experiment": exp_id, "versions": versions, "results": results},
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Full results (replies, settings, errors with tracebacks): {out}")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="Hugging Face model name or local folder")
    parser.add_argument("--models-file",
                        help="test every model listed in this file, one after another (see models.txt)")
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

    if args.models_file:
        ok = run_experiment(read_models_file(args.models_file), versions, args.save)
        raise SystemExit(0 if ok else 1)

    if not args.model:
        parser.error("give --model, --models-file, or set the MODEL_PATH environment variable")

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
