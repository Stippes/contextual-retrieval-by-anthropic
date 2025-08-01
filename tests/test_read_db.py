import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('.'))

from src.db.read_db import SemanticBM25Retriever


def test_missing_bm25_index(monkeypatch, tmp_path):
    monkeypatch.setenv("BASE_PATH", str(tmp_path))
    monkeypatch.setenv("VECTOR_DB_PATH", "vector")
    monkeypatch.setenv("BM25_DB_PATH", "missing")

    with pytest.raises(FileNotFoundError, match="BM25 index not found"):
        SemanticBM25Retriever(collection_name="test")
