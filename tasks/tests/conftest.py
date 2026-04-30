"""
conftest.py — executed before test collection.

Stubs heavy optional dependencies (langchain_chroma, langchain_core) that are
imported transitively by mas/memory/mas_memory/__init__.py but are not needed
for the unit tests in this directory.
"""

import sys
import os
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
