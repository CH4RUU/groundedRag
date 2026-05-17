"""
app/retrieval/reranker.py
─────────────────────────
Cross-Encoder Re-ranking using a local HuggingFace CrossEncoder.
Takes the initial Top-K from the hybrid retriever and rescores them
as query-document pairs to ensure ultra-high precision before feeding the LLM.

The CrossEncoder model is cached at module level so it is only loaded
once per process — not once per question.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from app.config import get_settings

settings = get_settings()

# ── Module-level singleton ─────────────────────────────────────────────────────
# Loaded once when the module is first imported. Subsequent calls reuse the
# already-loaded weights, saving ~7s of HuggingFace API calls per question.
_cross_encoder: HuggingFaceCrossEncoder | None = None


def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _cross_encoder


def get_reranking_compressor() -> CrossEncoderReranker:
    model = _get_cross_encoder()
    return CrossEncoderReranker(model=model, top_n=settings.rerank_top_n)

def get_reranking_retriever(base_retriever: BaseRetriever) -> ContextualCompressionRetriever:
    """
    Wraps the base (hybrid) retriever with a cached HuggingFace CrossEncoder.
    Only the top `rerank_top_n` documents are passed to the LLM.
    """
    compressor = get_reranking_compressor()

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
