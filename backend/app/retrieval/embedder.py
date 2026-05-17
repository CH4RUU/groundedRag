"""
app/retrieval/embedder.py
─────────────────────────
Cohere Embeddings wrapper.
Using 'embed-english-v3.0' which is industry standard and has a generous free tier.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached singleton of the embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
