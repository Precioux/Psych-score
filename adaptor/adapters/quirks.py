# adapters/quirks.py
# Small, per-model-type fixes for models whose own code needs extra setup before it can
# answer through the standard generate() path.
#
# The general adapter (hf_local.py) never mentions a specific model. Exceptions live only
# here, keyed by the "model_type" in the model's config.json, and every fix that was applied
# is recorded in info() and therefore in every log line.
#
# To support a new problematic model: write one function below and add it to QUIRKS.

import functools

from transformers import GenerationConfig, GenerationMixin


def _ensure_can_generate(module):
    """Old custom code may define a model that no longer gets generate() from transformers >= 4.50.
    Give it back by adding GenerationMixin to its class. Returns True if a change was made."""
    if module is None or isinstance(module, GenerationMixin):
        return False
    cls = module.__class__
    module.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
    if getattr(module, "generation_config", None) is None:
        module.generation_config = GenerationConfig.from_model_config(module.config)
    return True


def _accept_new_cache_objects(module):
    """Old custom code expects the cache ("past_key_values") as a tuple, or None on the first step.
    transformers >= 4.36 passes a Cache object instead (an empty one on the first step).
    Translate it before the old code sees it. Returns True if a change was made."""
    if module is None or getattr(module, "_legacy_cache_patch", False):
        return False
    original = module.prepare_inputs_for_generation

    # Keep the original's parameter names visible (functools.wraps + explicit arguments):
    # transformers inspects this signature, e.g. to check that inputs_embeds is supported.
    @functools.wraps(original)
    def prepare(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
            get_length = getattr(past_key_values, "get_seq_length", None)
            if get_length is not None and get_length() == 0:
                past_key_values = None                                   # first step: no cache yet
            elif hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()      # filled: old tuple format
        return original(input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds, **kwargs)

    module.prepare_inputs_for_generation = prepare
    module._legacy_cache_patch = True
    return True


def internvl_chat(model, tokenizer):
    """InternVL (OpenGVLab). Its own chat() normally does these two things before generating."""
    applied = []
    # 1. generate() asserts that the image-placeholder token id is known, even for text-only input.
    if getattr(model, "img_context_token_id", 0) is None:
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        applied.append("set img_context_token_id from <IMG_CONTEXT>")
    # 2. Its inner language model (InternLM2 / Qwen2 custom code) may lack generate().
    if _ensure_can_generate(getattr(model, "language_model", None)):
        applied.append("added GenerationMixin to language_model")
    # 3. Its inner language model expects the old tuple cache format.
    if _accept_new_cache_objects(getattr(model, "language_model", None)):
        applied.append("translated new Cache objects to the old cache format for language_model")
    return applied


QUIRKS = {
    "internvl_chat": internvl_chat,
}


def apply_quirks(model, tokenizer):
    """Apply the fix for this model's type, if there is one. Returns the list of changes made."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    fix = QUIRKS.get(model_type)
    return fix(model, tokenizer) if fix else []