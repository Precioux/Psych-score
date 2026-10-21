# adapters/base.py
# The one shape every model connection must follow, plus automatic logging.
#
# Every call to ask() - from any script, for any model, through any adapter - is appended
# to logs/<model name>/<date>.jsonl: the exact prompt, the reply (or the error), timing,
# and the adapter's settings. Adapters only implement _ask() and info(); logging is here,
# so no adapter can forget it.

import datetime
import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

# Default: the "logs" folder next to the "adapters" folder. Override with PSYCH_LOG_DIR.
DEFAULT_LOG_DIR = Path(os.environ.get("PSYCH_LOG_DIR",
                                      Path(__file__).resolve().parents[1] / "logs"))

_write_lock = threading.Lock()   # keeps lines intact if several threads log at once


class ModelAdapter(ABC):
    name: str = "unnamed"         # short label used in results and log folders
    log_dir: Path = DEFAULT_LOG_DIR

    # ---- what each adapter must implement ----
    @abstractmethod
    def _ask(self, system: str, user: str, max_new_tokens: int) -> str:
        """Actually send the messages to the model and return the reply text."""

    @abstractmethod
    def info(self) -> dict:
        """The settings actually used, saved with every logged call."""

    # ---- what the rest of the platform calls ----
    def ask(self, system: str, user: str, max_new_tokens: int = 512,
            context: dict | None = None) -> str:
        """Send the rules (system) and the questionnaire (user); return the reply.

        `context` is optional extra information to store with the call,
        e.g. {"scale": "BFI", "run": 3, "attempt": 0}.
        """
        record = {
            "call_id": uuid.uuid4().hex,
            "session_id": self._session_id(),
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": self.name,
            "context": context or {},
            "max_new_tokens": max_new_tokens,
            "system": system,
            "user": user,
        }
        start = time.time()
        try:
            reply = self._ask(system, user, max_new_tokens)
        except Exception as e:
            record.update(status="error", error=repr(e),
                          seconds=round(time.time() - start, 2), settings=self._safe_info())
            self._write_log(record)
            raise
        record.update(status="ok", reply=reply,
                      seconds=round(time.time() - start, 2), settings=self._safe_info())
        self._write_log(record)
        return reply

    # ---- logging helpers ----
    def _session_id(self) -> str:
        # one id per loaded model, so you can group the calls of one run of a script
        if not hasattr(self, "_session"):
            self._session = uuid.uuid4().hex[:12]
        return self._session

    def _safe_info(self) -> dict:
        try:
            return self.info()
        except Exception as e:  # never lose a log line because info() failed
            return {"info_error": repr(e)}

    def log_file(self) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.name)
        return Path(self.log_dir) / safe_name / f"{datetime.date.today().isoformat()}.jsonl"

    def _write_log(self, record: dict) -> None:
        path = self.log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
