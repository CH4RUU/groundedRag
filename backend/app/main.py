"""
app/main.py
───────────
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import HealthResponse
from app.api import ingest as ingest_router
from app.api import query as query_router
from app.retrieval.vector_store import check_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("RAG backend starting up (env=%s)", settings.environment)
    logger.info("Collection: %s", settings.qdrant_collection_name)
    yield
    logger.info("RAG backend shutting down")


app = FastAPI(
    title="Production-Grade RAG API",
    description="Hybrid retrieval (Dense + BM25) with Cohere Rerank and Gemini.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router.router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest_router.router, prefix="/api/v1", tags=["Ingest"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    qdrant_ok = await check_connection()
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        environment=settings.environment,
        qdrant_connected=qdrant_ok,
    )


@app.get("/", tags=["Root"])
async def root():
    return JSONResponse(
        {"message": "Production-Grade RAG API", "docs": "/docs", "health": "/health"}
    )
