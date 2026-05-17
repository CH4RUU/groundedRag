"""
app/observability/langfuse_handler.py
─────────────────────────────────────
Provides the Langfuse callback handler if configured.
Inject this into LangChain's `config={"callbacks": [handler]}`.
"""

import logging
from langfuse.callback import CallbackHandler
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_langfuse_handler(session_id: str = None) -> CallbackHandler | None:
    """
    Returns a Langfuse CallbackHandler if credentials exist.
    If session_id is provided, traces will be grouped in the UI.
    """
    if not settings.langfuse_enabled:
        return None

    try:
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=session_id,
            tags=[settings.environment, "rag-pipeline"],
        )
        return handler
    except Exception as e:
        logger.warning("Failed to initialize Langfuse handler: %s", e)
        return None
