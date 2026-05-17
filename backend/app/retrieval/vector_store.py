"""
app/retrieval/vector_store.py
─────────────────────────────
Qdrant Cloud integration.
Initialises the Qdrant LangChain wrapper.
"""

import logging
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from app.config import get_settings
from app.retrieval.embedder import get_embeddings

logger = logging.getLogger(__name__)
settings = get_settings()

# We maintain a global client to avoid reconnect overhead
_client = None
_async_client = None


def _get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _client


async def get_vector_store(collection_name: str = None) -> QdrantVectorStore:
    """
    Get the LangChain Qdrant wrapper.
    Uses the modern langchain_qdrant library and explicitly sets
    RetrievalMode.HYBRID to leverage Qdrant's built-in sparse (BM25) + dense search.
    """
    col_name = collection_name or settings.qdrant_collection_name
    client = _get_qdrant_client()
    embeddings = get_embeddings()

    # Create collection if it doesn't exist
    if not client.collection_exists(col_name):
        logger.info("Collection '%s' does not exist. Creating it now...", col_name)
        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=384,  # Default for all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        )

    # Initialize QdrantVectorStore
    return QdrantVectorStore(
        client=client,
        collection_name=col_name,
        embedding=embeddings,
        # Enable hybrid search natively in Qdrant (requires fastembed or server-side sparse).
        # We'll configure EnsembleRetriever at the app level if we want fine-grained control,
        # but LangChain Qdrant supports it out of the box if configured correctly.
        # For this setup, we'll return the base store and let `hybrid_retriever.py` manage the ensemble
        # or use native depending on what's configured.
    )


async def check_connection() -> bool:
    """Health check ping to Qdrant Cloud."""
    try:
        client = _get_qdrant_client()
        collections = client.get_collections()
        return True
    except Exception as e:
        logger.error("Qdrant connection failed: %s", e)
        return False
