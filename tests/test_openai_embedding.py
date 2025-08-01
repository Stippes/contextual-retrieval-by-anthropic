import sys, os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
from src.openai_client import OpenAIEmbedding, get_embeddings_async


def test_openai_embedding_not_abstract():
    assert OpenAIEmbedding.__abstractmethods__ == set()


class DummyAClient:
    class embeddings:
        @staticmethod
        async def create(model, input):
            return type('Resp', (), {'data': [type('X', (), {'embedding': [0.0]})()]})()


def test_get_embeddings_async(monkeypatch):
    monkeypatch.setattr('src.openai_client._get_async_client', lambda: DummyAClient())
    result = asyncio.run(get_embeddings_async(['x']))
    assert result == [[0.0]]
