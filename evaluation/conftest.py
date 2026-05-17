"""
evaluation/conftest.py
──────────────────────
Shared pytest fixtures for evaluation tests.
"""
import sys
from pathlib import Path

import pytest

# Ensure backend app module is importable
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def golden_dataset():
    """Load the golden Q&A dataset as a fixture."""
    import json
    path = Path(__file__).parent / "golden_dataset.json"
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture
def sample_question():
    return "What is LangChain Expression Language (LCEL)?"
