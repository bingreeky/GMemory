"""
conftest.py — executed before test collection.

Stubs heavy optional dependencies (langchain_chroma, langchain_core) that are
imported transitively by mas/memory/mas_memory/__init__.py but are not needed
for the unit tests in this directory.

Also tees all stdout/stderr output to a timestamped .txt file in tasks/tests/logs/
so that a full record of every run (including -s print output) is preserved.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Set required env vars before importing mas (mas/__init__.py reads them).
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:9999")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


def _stub_module(name: str):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


for _mod in (
    "langchain_chroma",
    "langchain_core",
    "langchain_core.documents",
    "finch",
):
    _stub_module(_mod)


# ── output logging ────────────────────────────────────────────────────────────

class _TeeWriter:
    """Writes to multiple streams simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for s in self._streams:
            s.write(text)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def pytest_configure(config):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.txt"

    log_file = log_path.open("w", encoding="utf-8")
    config._tee_log_file = log_file
    config._tee_log_path = log_path

    config._orig_stdout = sys.stdout
    config._orig_stderr = sys.stderr
    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)


def pytest_unconfigure(config):
    if hasattr(config, "_orig_stdout"):
        sys.stdout = config._orig_stdout
    if hasattr(config, "_orig_stderr"):
        sys.stderr = config._orig_stderr
    if hasattr(config, "_tee_log_file"):
        config._tee_log_file.close()
        print(f"\nTest output saved to: {config._tee_log_path}")
