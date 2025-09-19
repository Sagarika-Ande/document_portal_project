# tests/test_unit_cases.py
#pytest tests/test_unit_cases.py -v
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint
from src.document_ingestion.data_ingestion import FileProcessor

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


class DummyFile:
    """Mock uploaded file with .name and .read()"""
    def __init__(self, name, content: str):
        self.name = name
        self._content = content.encode()

    def read(self):
        return io.BytesIO(self._content).read()


def test_save_file(tmp_path):
    processor = FileProcessor(base_dir=tmp_path, session_id="test_session")
    uploaded_file = DummyFile("sample.txt", "Hello Test")

    response_path = processor.save_file(uploaded_file)

    assert response_path.exists()
    assert response_path.read_text() == "Hello Test"


def test_read_file(tmp_path):
    processor = FileProcessor(base_dir=tmp_path, session_id="test_session")
    file_path = processor.session_path / "example.txt"

    # Create file in session path
    file_path.write_text("Unit testing FileProcessor")

    response_text = processor.read_file(file_path)

    assert response_text == "Unit testing FileProcessor"
