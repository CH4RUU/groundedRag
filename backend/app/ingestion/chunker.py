"""
app/ingestion/chunker.py
────────────────────────
Splits documents into smaller overlapping chunks.
Uses tiktoken for accurate token-based lengths (better for LLMs than char counts).
Every chunk receives a unique UUID so it can be cited later.
"""

import uuid
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initialize splitter with settings from config."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # Standard for OpenAI/Gemini
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents and assign a unique chunk_id to each."""
    if not documents:
        return []

    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        # Generate a unique deterministic ID based on the text + metadata if desired,
        # but a UUID is safer to guarantee uniqueness for citations.
        chunk.metadata["chunk_id"] = f"chunk_{uuid.uuid4().hex[:8]}"
        # Ensure 'source' is string (sometimes it's a URL)
        chunk.metadata["source"] = str(chunk.metadata.get("source", "unknown"))

    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
