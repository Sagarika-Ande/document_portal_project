# tests/test_unit_cases.py
#pytest tests/test_unit_cases.py -v

import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


def test_document_upload_success():
    # Assume you have an endpoint like /upload for document submission
    # And a dummy file to upload for testing
    with open("tests/dummy_document.txt", "rb") as f:
        response = client.post("/upload", files={"file": ("dummy_document.txt", f, "text/plain")})
    assert response.status_code == 200
    assert "Document uploaded and processed successfully" in response.json().get("message", "")
    # Further assertions could check if the document ID is returned or if it's visible in a document list


def test_chat_single_document_query():
    # First, ensure a document is processed or mock its presence
    # Assume an endpoint like /chat that takes a query and optionally a document_id
    response = client.post("/chat", json={"query": "What is the main topic of the document?", "document_id": "doc_123"})
    assert response.status_code == 200
    assert "response" in response.json()
    assert len(response.json()["response"]) > 0 # Check that a non-empty response is returned
    # More specific checks: assert "RAG system" in response.json()["response"] if that's expected