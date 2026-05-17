"""
app/rag/prompt_registry.py
──────────────────────────
Loads version-controlled prompts from YAML files.
Treats prompts as core infrastructure rather than hardcoded strings.
"""

import logging
from pathlib import Path
import yaml
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(version: str = None) -> ChatPromptTemplate:
    """
    Loads the specified prompt version from the prompts directory.
    If no version is specified, uses the one from settings.
    """
    prompt_version = version or settings.prompt_version
    file_path = PROMPTS_DIR / f"rag_system_{prompt_version}.yaml"

    if not file_path.exists():
        logger.error("Prompt file not found: %s", file_path)
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    system_message = config.get("system", "")
    human_message = config.get("human", "{question}")

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(human_message)
    ])
