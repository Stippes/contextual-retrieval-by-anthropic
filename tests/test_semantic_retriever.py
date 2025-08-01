import os, sys
sys.path.insert(0, os.path.abspath('.'))

from src.db.read_db import SemanticBM25Retriever
from src.openai_client import OpenAIEmbedding


def test_semantic_retriever_init(monkeypatch):
    class DummyCollection:
        pass

    class DummyPersistentClient:
        def __init__(self, path):
            pass
        def get_or_create_collection(self, name):
            return DummyCollection()

    class DummyVectorStore:
        pass

    class DummyIndex:
        def as_retriever(self):
            class DR:
                def retrieve(self, *args, **kwargs):
                    return []
            return DR()

    class DummyBM25:
        def retrieve(self, *args, **kwargs):
            return []

    monkeypatch.setattr('src.db.read_db.chromadb.PersistentClient', lambda path: DummyPersistentClient(path))
    monkeypatch.setattr('src.db.read_db.ChromaVectorStore', lambda chroma_collection=None: DummyVectorStore())
    monkeypatch.setattr('src.db.read_db.VectorStoreIndex.from_vector_store', lambda *args, **kwargs: DummyIndex())
    monkeypatch.setattr('src.db.read_db.BM25Retriever.from_persist_dir', lambda path: DummyBM25())

    retriever = SemanticBM25Retriever(collection_name='test')
    assert isinstance(retriever._embed_model, OpenAIEmbedding)
