# adapters/hf_local.py
# Route 1: models stored in Hugging Face format, loaded on this machine with `transformers`.
# Works for text-only (Llama, Qwen2, Mistral, ...) and vision-language (Qwen2-VL, Qwen3-VL,
# Gemma 3, ...) models, and - with trust_remote_code=True - models that ship their own code
# (e.g. InternVL).

import torch
import transformers
from transformers import (AutoProcessor, AutoModel, AutoModelForCausalLM,
                          PreTrainedTokenizerBase)

try:  # only exists in newer transformers versions
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from .base import ModelAdapter

# End-of-turn markers used by common chat formats. If the tokenizer knows one of them,
# generation also stops there (some custom-code models don't declare it themselves).
END_OF_TURN_TOKENS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>"]


class HFLocalAdapter(ModelAdapter):
    def __init__(self, model_path, name=None, dtype="auto", trust_remote_code=False):
        self.model_path = model_path
        self.name = name or model_path.rstrip("/").split("/")[-1]
        self.trust_remote_code = trust_remote_code

        # The processor turns text into numbers (plain tokenizer for text models,
        # tokenizer + image processor for vision-language models).
        self.processor = AutoProcessor.from_pretrained(model_path,
                                                       trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.is_multimodal = not isinstance(self.processor, PreTrainedTokenizerBase)

        self.loader_name = None          # filled in by _load: which loader worked
        self.model = self._load(model_path, dtype)
        self.model.eval()
        self.stop_ids = self._stop_ids()

    # ---- loading: try each model family until one accepts this model ----
    def _load(self, model_path, dtype):
        loaders = [AutoModelForCausalLM, AutoModelForImageTextToText]
        if self.trust_remote_code:
            # Custom-code models (InternVL, ...) often register only under AutoModel.
            # Not used otherwise: for standard models AutoModel gives a model that can't generate.
            loaders.append(AutoModel)
        errors = []
        for loader in loaders:
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(model_path, dtype=dtype, device_map="auto",
                                               trust_remote_code=self.trust_remote_code)
                self.loader_name = loader.__name__
                return model
            except ValueError as e:
                errors.append(f"{loader.__name__}: {str(e)[:200]}")
        raise ValueError("No loader could open this model:\n" + "\n".join(errors))

    def _stop_ids(self):
        """All token ids that should end the answer: the model's own + known end-of-turn markers."""
        ids = set()
        configured = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        for value in (configured, self.tokenizer.eos_token_id):
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, (list, tuple)):
                ids.update(v for v in value if isinstance(v, int))
        unk = self.tokenizer.unk_token_id
        vocab = self.tokenizer.get_vocab()
        for token in END_OF_TURN_TOKENS:
            token_id = vocab.get(token)
            if token_id is not None and token_id != unk:
                ids.add(token_id)
        return sorted(ids)

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
            pad_id = self.stop_ids[0] if self.stop_ids else None

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                            # greedy = temperature 0
                temperature=None, top_p=None, top_k=None,   # ignore the model's sampling defaults
                pad_token_id=pad_id,
                eos_token_id=self.stop_ids or None,
            )

        # Most models return prompt + answer; some custom-code models (e.g. InternVL)
        # return only the answer. Cut the prompt off only if it is really there.
        prompt_ids = inputs["input_ids"][0]
        n = prompt_ids.shape[-1]
        sequence = output[0]
        if sequence.shape[-1] >= n and torch.equal(sequence[:n].to(prompt_ids.device), prompt_ids):
            sequence = sequence[n:]
        return self.tokenizer.decode(sequence, skip_special_tokens=True).strip()

    def info(self):
        return {
            "adapter": "hf_local",
            "name": self.name,
            "model_path": self.model_path,
            "loader": self.loader_name,
            "multimodal": self.is_multimodal,
            "trust_remote_code": self.trust_remote_code,
            "dtype": str(self.model.dtype),
            "device": str(self.model.device),
            "decoding": "greedy (do_sample=False)",
            "stop_token_ids": self.stop_ids,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
