"""
app/rag/chain.py
────────────────
Assembles the LCEL (LangChain Expression Language) pipeline.
Uses `with_structured_output` to strictly bind the LLM to the Pydantic schema,
preventing JSON hallucinations.
LLM backend: Cerebras (llama3.1-8b) — fast inference via OpenAI-compatible API.
"""

import time
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.config import get_settings
from app.models import RAGResponse, ChunkMetadata
from app.rag.prompt_registry import load_prompt
from app.retrieval.hybrid_retriever import get_hybrid_retriever
from app.retrieval.reranker import get_reranking_compressor
from app.retrieval.embedder import get_embeddings
import math

logger = logging.getLogger(__name__)
settings = get_settings()


def format_context(documents: List[Document]) -> str:
    """Formats chunks into a string for the prompt."""
    formatted = []
    for i, doc in enumerate(documents):
        chunk_id = doc.metadata.get("chunk_id", f"unknown-{i}")
        source = doc.metadata.get("source", "unknown")
        # Include chunk_id explicitly so the LLM can cite it
        formatted.append(f"--- Chunk ID: {chunk_id} | Source: {source} ---\n{doc.page_content}")
    return "\n\n".join(formatted)


async def execute_rag_pipeline(
    question: str,
    collection_name: str = None,
    callbacks: list = None
) -> Dict[str, Any]:
    """
    Executes the full retrieval and generation pipeline.
    Returns a dictionary containing the structured response, retrieved chunks, and latency.
    """
    start_time = time.time()

    # Extract Langfuse trace URL if available
    langfuse_url = None
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, "get_trace_url"):
                langfuse_url = cb.get_trace_url()

    # 1. Embedding Timing (mocking the exact call done inside hybrid)
    t_emb = time.time()
    get_embeddings().embed_query(question)
    embedding_ms = (time.time() - t_emb) * 1000

    # 2. Retrieval Setup & Base Retrieval
    t0 = time.time()
    base_retriever = await get_hybrid_retriever(collection_name)
    raw_base_docs = await base_retriever.ainvoke(question, config={"callbacks": callbacks})
    
    # Advanced Deduplication: Remove exact duplicate chunks from the database
    # to prevent identical 9.2% scores that look like bugs
    seen_content = set()
    base_docs = []
    for doc in raw_base_docs:
        # Normalize slightly to catch near-duplicates
        content_hash = doc.page_content.strip()[:100]
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            base_docs.append(doc)
            
    hybrid_search_ms = (time.time() - t0) * 1000
    hybrid_search_ms = max(0, hybrid_search_ms - embedding_ms) # isolate search from embedding

    # 3. Rerank
    t1 = time.time()
    logger.info("Retrieving and re-ranking chunks for query: %s", question)
    
    from app.retrieval.reranker import _get_cross_encoder
    cross_encoder = _get_cross_encoder()
    
    # Score pairs manually to guarantee metadata isn't dropped by LangChain
    pairs = [[question, doc.page_content] for doc in base_docs]
    scores = cross_encoder.score(pairs)
    
    # Attach raw logits directly to the doc metadata and sort
    for doc, score in zip(base_docs, scores):
        doc.metadata["explicit_cross_encoder_score"] = float(score)
    
    base_docs.sort(key=lambda x: x.metadata["explicit_cross_encoder_score"], reverse=True)
    docs = base_docs[:settings.rerank_top_n]
    reranking_ms = (time.time() - t1) * 1000

    # 4. Prepare Context
    context_str = format_context(docs)

    # 5. LLM Setup (Cerebras — 1M tokens/day, OpenAI-compatible)
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.cerebras_api_key,
        base_url="https://api.cerebras.ai/v1",
    )
    # Force output to exactly match our Pydantic schema using json_mode to prevent tool-calling parsing errors
    structured_llm = llm.with_structured_output(RAGResponse, method="json_mode")

    # 6. Prompt setup
    prompt = load_prompt()
    chain = prompt | structured_llm

    # 7. Generate Response
    t2 = time.time()
    logger.info("Generating response with structured output...")
    rag_response: RAGResponse = await chain.ainvoke(
        {"context": context_str, "question": question},
        config={"callbacks": callbacks}
    )
    llm_generation_ms = (time.time() - t2) * 1000

    if rag_response is None:
        raise ValueError("LLM returned None instead of a structured response. Parsing failed.")

    # 8. Package metadata for the client
    latency_ms = (time.time() - start_time) * 1000

    retrieved_chunks = []
    for i, doc in enumerate(docs):
        # We explicitly set this during our manual reranking phase above
        raw_score = doc.metadata.get("explicit_cross_encoder_score", 0.0)

        # MS-MARCO typically outputs logits between -10 and +10. We apply Sigmoid to get probability.
        if raw_score == 0.0:
            confidence_score = 0.0
        elif raw_score < -100 or raw_score > 100:
            confidence_score = 0.0 # safety catch
        else:
            confidence_score = 1 / (1 + math.exp(-raw_score))
        
        retrieved_chunks.append(
            ChunkMetadata(
                chunk_id=doc.metadata.get("chunk_id", f"chunk_{i}"),
                source=doc.metadata.get("source", "unknown"),
                content_preview=doc.page_content[:200] + "...",
                score=confidence_score,
                retrieval_modality="Hybrid (BM25 + Dense)",
            )
        )

    sources_list = list(set([doc.source for doc in retrieved_chunks]))

    return {
        "answer": rag_response.answer,
        "citations": rag_response.citations,
        "sources": sources_list,
        "refusal": rag_response.refusal,
        "confidence": rag_response.confidence,
        "retrieved_chunks": retrieved_chunks,
        "full_contexts": [doc.page_content for doc in docs],  # Full text for Ragas evaluation
        "latency_ms": latency_ms,
        "latency_breakdown": {
            "Embedding": round(embedding_ms, 1),
            "Hybrid Search": round(hybrid_search_ms, 1),
            "Reranking": round(reranking_ms, 1),
            "LLM Generation": round(llm_generation_ms, 1)
        },
        "langfuse_url": langfuse_url,
        "prompt_version": settings.prompt_version,
    }
