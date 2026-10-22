# adapters/quirks.py
# Small, per-model-type fixes for models whose own code needs extra setup before it can
# answer through the standard generate() path.
#
# The general adapter (hf_local.py) never mentions a specific model. Exceptions live only
# here, keyed by the "model_type" in the model's config.json, and every fix that was applied
# is recorded in info() and therefore in every log line.
#
# To support a new problematic model: write one function below and add it to QUIRKS.

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
    return applied


QUIRKS = {
    "internvl_chat": internvl_chat,
}


def apply_quirks(model, tokenizer):
    """Apply the fix for this model's type, if there is one. Returns the list of changes made."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    fix = QUIRKS.get(model_type)
    return fix(model, tokenizer) if fix else []
