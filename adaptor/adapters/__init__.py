# Only the interface is imported here. Each adapter is imported on its own
# (e.g. `from adapters.hf_local import HFLocalAdapter`), so an API-only setup
# never needs torch or transformers installed.
from .base import ModelAdapter

__all__ = ["ModelAdapter"]
