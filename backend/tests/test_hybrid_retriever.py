import pytest
from unittest.mock import patch, MagicMock
from app.retrieval.hybrid_retriever import get_hybrid_retriever, initialize_bm25_from_docs
from langchain_core.documents import Document

@pytest.mark.asyncio
@patch("app.retrieval.hybrid_retriever.get_vector_store")
async def test_hybrid_retriever_initialization(mock_get_vector_store):
    # Mock the dense retriever
    mock_store = MagicMock()
    mock_dense_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_dense_retriever
    mock_get_vector_store.return_value = mock_store

    # Initialize BM25
    docs = [Document(page_content="test")]
    await initialize_bm25_from_docs(docs)

    # Get the ensemble
    retriever = await get_hybrid_retriever()
    
    assert retriever is not None
    assert len(retriever.retrievers) == 2
    assert retriever.weights == [0.5, 0.5]
