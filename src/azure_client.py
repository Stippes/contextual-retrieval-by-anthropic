import os
from typing import List
from openai import AzureOpenAI

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")

def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


def chat_completion(prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def get_embeddings(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    response = client.embeddings.create(model=AZURE_DEPLOYMENT_NAME, input=texts)
    return [d.embedding for d in response.data]

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

class AzureEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> Embedding:
        return get_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> Embedding:
        return get_embeddings([query])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return get_embeddings(texts)
