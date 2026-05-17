"""
app/api/ingest.py
─────────────────
POST /ingest endpoint to load URLs into the vector store dynamically.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models import IngestRequest, IngestResponse
from app.ingestion.loader import load_urls
from app.ingestion.ingest import ingest_pipeline
from app.retrieval.hybrid_retriever import initialize_bm25_from_docs

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_urls(request: IngestRequest) -> IngestResponse:
    """
    Scrape given URLs, chunk them, embed, and upsert to Qdrant.
    Also re-initializes the BM25 index with the new docs.
    """
    try:
        # Load from web
        docs = load_urls(request.urls)
        
        # In real production, BM25 should be persistent. 
        # Here we just initialize it in-memory for the demo.
        await initialize_bm25_from_docs(docs)

        # Ingest
        chunks_added = await ingest_pipeline(docs, request.collection_name)

        return IngestResponse(
            status="success",
            chunks_added=chunks_added,
            collection_name=request.collection_name or "default",
            message=f"Successfully ingested {len(docs)} documents."
        )

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
