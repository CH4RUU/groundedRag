"""
app/retrieval/hybrid_retriever.py
─────────────────────────────────
Combines Dense Retrieval (Vector) with Sparse Retrieval (BM25) using LangChain's EnsembleRetriever.
BM25 excels at exact keyword matches, while Dense excels at semantic meaning.
EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to merge the results.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List

from app.config import get_settings
from app.retrieval.vector_store import get_vector_store

settings = get_settings()

# We cache the BM25 retriever in memory for this example.
# In a true distributed system, you'd use Qdrant's native sparse vectors
# or Elasticsearch, but BM25Retriever is great for showcasing the Ensemble logic.
_bm25_retriever_cache = None


async def initialize_bm25_from_docs(documents: List[Document]):
    """Call this during ingestion to build the BM25 index."""
    global _bm25_retriever_cache
    if not documents:
        return
    _bm25_retriever_cache = BM25Retriever.from_documents(documents)
    _bm25_retriever_cache.k = settings.top_k


async def get_hybrid_retriever(collection_name: str = None) -> EnsembleRetriever:
    """
    Returns an EnsembleRetriever combining Qdrant (dense) and BM25 (sparse).
    Weights are set to 0.5 / 0.5 as requested by the user for balanced RRF.
    """
    vector_store = await get_vector_store(collection_name)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k})

    global _bm25_retriever_cache

    if _bm25_retriever_cache is None:
        # Fallback to pure dense if BM25 hasn't been initialized
        # (e.g., server restart without persistent BM25 storage)
        return dense_retriever

    # Reciprocal Rank Fusion formula is applied automatically by EnsembleRetriever:
    # score = 1 / (k + rank), where k=60 by default.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, _bm25_retriever_cache],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever
