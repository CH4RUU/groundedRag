import pytest
from langchain_core.documents import Document
from app.ingestion.chunker import chunk_documents
from app.config import get_settings

settings = get_settings()

def test_chunk_documents_assigns_uuids():
    docs = [
        Document(page_content="This is a test document. " * 100, metadata={"source": "test.txt"})
    ]
    chunks = chunk_documents(docs)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("chunk_")
        assert chunk.metadata["source"] == "test.txt"

def test_chunk_size_limits():
    docs = [
        Document(page_content="A" * 5000, metadata={"source": "test.txt"})
    ]
    chunks = chunk_documents(docs)
    
    # Check that chunks respect the rough token limit
    # (Since it's tiktoken, char length will vary, but should be bounded)
    for chunk in chunks:
        assert len(chunk.page_content) > 0
        # rough heuristic: 1 token ~ 4 chars
        assert len(chunk.page_content) <= settings.chunk_size * 5 
