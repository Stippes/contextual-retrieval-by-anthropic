from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from .fusion import reciprocal_rank_fusion
import chromadb
import Stemmer
from typing import List
import os
from dotenv import load_dotenv
from src.logging_config import get_logger

if os.getenv("OPENAI_API_KEY"):
    from src.openai_client import OpenAIEmbedding as EmbeddingModel
else:
    from src.azure_client import AzureEmbedding as EmbeddingModel

load_dotenv()

logger = get_logger(__name__)


class SemanticBM25Retriever(BaseRetriever):
    def __init__(self, collection_name: str = "default", mode: str = "OR") -> None:

        self._mode = mode

        # Path to database directories
        BASE_PATH = os.getenv("BASE_PATH", "")
        VECTOR_DB_PATH = os.path.join(BASE_PATH, os.getenv("VECTOR_DB_PATH", ""))
        BM25_DB_PATH = os.path.join(BASE_PATH, os.getenv("BM25_DB_PATH", ""))

        try:
            # Embedding Model
            self._embed_model = EmbeddingModel()

            # Read stored Vector Database
            self._vectordb = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            _chroma_collection = self._vectordb.get_or_create_collection(
                collection_name
            )
            self._vector_store = ChromaVectorStore(chroma_collection=_chroma_collection)
            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store,
                embed_model=self._embed_model,
            )

            self._chromadb_retriever = self._index.as_retriever()

            # Read stored BM25 Database
            self._bm25_retriever = BM25Retriever.from_persist_dir(BM25_DB_PATH)
        except Exception:
            logger.exception("Failed to initialize retrievers")
            raise

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.info(f"Querying database for: {query_bundle.query_str}")
        try:
            # Retrieving Nodes from Database
            vector_nodes = self._chromadb_retriever.retrieve(query_bundle)
            bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

            fused_nodes = reciprocal_rank_fusion(vector_nodes, bm25_nodes)
        except Exception:
            logger.exception("Error retrieving documents")
            raise

        filenames = []
        for n in fused_nodes:
            meta = getattr(n.node, "metadata", {}) or {}
            fname = meta.get("file_name") or meta.get("file_path")
            if fname:
                filenames.append(os.path.basename(fname))
        if filenames:
            logger.info("Documents accessed: %s", ", ".join(filenames))

        if self._mode == "AND":
            vector_ids = {n.node.node_id for n in vector_nodes}
            bm25_ids = {n.node.node_id for n in bm25_nodes}
            valid_ids = vector_ids.intersection(bm25_ids)
            fused_nodes = [n for n in fused_nodes if n.node.node_id in valid_ids]

        return fused_nodes


if __name__ == "__main__":

    db = SemanticBM25Retriever(collection_name="cook_book")

    res = db.retrieve("List of all sandwich recipes.")

    print(len(res))
