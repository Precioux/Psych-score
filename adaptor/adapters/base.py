# adapters/base.py
# The one shape every model connection must follow.
# The rest of the platform (questionnaires, parser, scoring) only ever calls ask() and info(),
# so it never needs to know whether the model is local, on a server, or behind an API.

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    name: str = "unnamed"   # short label used in results and file names

    @abstractmethod
    def ask(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        """Send the rules (system) and the questionnaire (user); return the reply as text."""

    @abstractmethod
    def info(self) -> dict:
        """The settings actually used, saved next to every result so runs can be reproduced."""
