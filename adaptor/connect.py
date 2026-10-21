# connect.py
# Model-agnostic connection to any local Hugging Face chat model,
# text-only (Llama, Qwen2.5, Mistral, ...) or vision-language (Qwen2-VL, Gemma 3, ...).
#
# Usage:
#   python connect.py --model Qwen/Qwen2-VL-7B-Instruct
#   python connect.py --model meta-llama/Llama-3.1-8B-Instruct
#   python connect.py --model /path/to/any/local/model/folder

# ---- Part 1: imports ----
import argparse
import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedTokenizerBase

try:  # only exists in newer transformers versions
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


# ---- Part 2: one class that wraps ANY model ----
class LocalModel:
    def __init__(self, model_path, dtype="auto"):
        # 2a. The processor turns text into numbers. For text-only models it is just the
        #     tokenizer; for vision-language models it is a tokenizer + image processor.
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.is_multimodal = not isinstance(self.processor, PreTrainedTokenizerBase)

        # 2b. Load the model with whichever loader accepts it
        self.model = self._load(model_path, dtype)
        self.model.eval()

    def _load(self, model_path, dtype):
        loaders = [AutoModelForCausalLM, AutoModelForImageTextToText]
        errors = []
        for loader in loaders:
            if loader is None:
                continue
            try:
                return loader.from_pretrained(model_path, dtype=dtype, device_map="auto")
            except ValueError as e:  # "Unrecognized configuration class ..." -> try the next one
                errors.append(f"{loader.__name__}: {str(e)[:200]}")
        raise ValueError("No loader could open this model:\n" + "\n".join(errors))

    # ---- Part 3: build the conversation in the shape this model expects ----
    def _messages(self, system, user):
        if self.is_multimodal:
            wrap = lambda text: [{"type": "text", "text": text}]   # VL models: list of parts
        else:
            wrap = lambda text: text                                 # text models: plain string
        messages = []
        if system:
            messages.append({"role": "system", "content": wrap(system)})
        messages.append({"role": "user", "content": wrap(user)})
        return messages

    def _encode(self, system, user):
        settings = dict(add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt")
        try:
            return self.processor.apply_chat_template(self._messages(system, user), **settings)
        except Exception:
            # Some models' templates reject a "system" role: put the rules in the user turn.
            merged = f"{system}\n\n{user}" if system else user
            return self.processor.apply_chat_template(self._messages(None, merged), **settings)

    # ---- Part 4: the connection function the framework calls ----
    def ask(self, system, user, max_new_tokens=512):
        inputs = self._encode(system, user).to(self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                   # greedy = temperature 0
                temperature=None, top_p=None, top_k=None,  # ignore model's sampling defaults
                pad_token_id=pad_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        answer_tokens = output[0][prompt_length:]
        return self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


# ---- Part 5: test with two questionnaire items ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="Hugging Face model name or local folder")
    args = parser.parse_args()
    if not args.model:
        parser.error("give --model or set the MODEL_PATH environment variable")

    print(f"Loading {args.model} ... (this can take a minute)")
    llm = LocalModel(args.model)
    print("Model loaded on:", llm.model.device, "| multimodal:", llm.is_multimodal)

    system = ('You are a helpful assistant who can only reply numbers from 1 to 5. '
              'Format: "statement index: score".')
    user = ("Rate how well each statement describes you. "
            "1 = disagree strongly, 2 = disagree a little, 3 = neutral, "
            "4 = agree a little, 5 = agree strongly.\n"
            "1. I see myself as someone who is talkative.\n"
            "2. I see myself as someone who is reserved.")

    print("Model reply:")
    print(llm.ask(system, user))