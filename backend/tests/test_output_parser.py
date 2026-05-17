import pytest
from app.models import RAGResponse
from pydantic import ValidationError

def test_rag_response_valid_schema():
    # Should pass
    resp = RAGResponse(
        answer="LangChain is a framework.",
        citations=["chunk_123"],
        confidence=0.9
    )
    assert resp.answer == "LangChain is a framework."
    assert len(resp.citations) == 1

def test_rag_response_refusal():
    # Should pass
    resp = RAGResponse(
        answer="",
        citations=[],
        refusal="I cannot answer this based on the context.",
        confidence=0.1
    )
    assert resp.refusal is not None

def test_rag_response_confidence_bounds():
    # Should fail due to confidence > 1.0
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="Test",
            citations=[],
            confidence=1.5
        )
