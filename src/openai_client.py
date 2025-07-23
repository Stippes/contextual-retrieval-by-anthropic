import os
from typing import List
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")


def _get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def chat_completion(prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def get_embeddings(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]


from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding


class OpenAIEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> Embedding:
        return get_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> Embedding:
        return get_embeddings([query])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return get_embeddings(texts)
