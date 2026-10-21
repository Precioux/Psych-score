# adapters/hf_local.py
# Route 1: models stored in Hugging Face format, loaded on this machine with `transformers`.
# Works for text-only (Llama, Qwen2.5, Mistral, ...) and vision-language (Qwen2-VL, Qwen3-VL, ...) models.
# The loading logic is unchanged from connect.py; it is only moved under the ModelAdapter interface.

import torch
import transformers
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedTokenizerBase

try:  # only exists in newer transformers versions
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from .base import ModelAdapter


class HFLocalAdapter(ModelAdapter):
    def __init__(self, model_path, name=None, dtype="auto"):
        self.model_path = model_path
        self.name = name or model_path.rstrip("/").split("/")[-1]

        # The processor turns text into numbers (plain tokenizer for text models,
        # tokenizer + image processor for vision-language models).
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.is_multimodal = not isinstance(self.processor, PreTrainedTokenizerBase)

        self.loader_name = None          # filled in by _load: which loader worked
        self.model = self._load(model_path, dtype)
        self.model.eval()

    # ---- loading: try each model family until one accepts this model ----
    def _load(self, model_path, dtype):
        loaders = [AutoModelForCausalLM, AutoModelForImageTextToText]
        errors = []
        for loader in loaders:
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(model_path, dtype=dtype, device_map="auto")
                self.loader_name = loader.__name__
                return model
            except ValueError as e:
                errors.append(f"{loader.__name__}: {str(e)[:200]}")
        raise ValueError("No loader could open this model:\n" + "\n".join(errors))

    # ---- build the conversation in the shape this model expects ----
    def _messages(self, system, user):
        if self.is_multimodal:
            wrap = lambda text: [{"type": "text", "text": text}]
        else:
            wrap = lambda text: text
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
            # Some templates reject a "system" role: put the rules in the user turn.
            merged = f"{system}\n\n{user}" if system else user
            return self.processor.apply_chat_template(self._messages(None, merged), **settings)

    # ---- the interface: sending (logging is done by ModelAdapter.ask) ----
    def _ask(self, system, user, max_new_tokens):
        inputs = self._encode(system, user).to(self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                            # greedy = temperature 0
                temperature=None, top_p=None, top_k=None,   # ignore the model's sampling defaults
                pad_token_id=pad_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        answer_tokens = output[0][prompt_length:]
        return self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    def info(self):
        return {
            "adapter": "hf_local",
            "name": self.name,
            "model_path": self.model_path,
            "loader": self.loader_name,
            "multimodal": self.is_multimodal,
            "dtype": str(self.model.dtype),
            "device": str(self.model.device),
            "decoding": "greedy (do_sample=False)",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
