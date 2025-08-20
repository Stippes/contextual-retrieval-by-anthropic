import os
from dotenv import load_dotenv
import tiktoken

from llama_index.core.schema import TextNode

from src.openai_client import chat_completion
from src.ingest.chunking import chunk_elements
from src.extractors import load_documents

from .save_vectordb import save_chromadb
from .save_bm25 import save_BM25

load_dotenv()

def create_and_save_db(
        data_dir: str,
        collection_name: str,
        save_dir: str,
        db_name: str = "default",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_document_tokens: int = 2048,
        context_window: int = 8192,
    ) -> None:
    """
    Ingests documents, chunks them, generates contextualized chunks, and saves both
    a vector DB (Chroma via LlamaIndex) and a BM25 index.

    Notes:
    - Metadata is flattened to scalars/strings to satisfy vector store constraints.
    - The contextual prompt is a collapsed single-line string to avoid indentation issues.
    """

    # ---------------------------
    # Helpers
    # ---------------------------
    def _flat(md):
        """Flatten metadata so all values are (str|int|float|None)."""
        import json
        out = {}
        for k, v in (md or {}).items():
            if isinstance(v, (str, int, float)) or v is None:
                out[k] = v
            elif isinstance(v, bool):
                out[k] = int(v)  # or str(v)
            elif isinstance(v, (list, tuple, set)):
                out[k] = ",".join(map(str, v))
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        return out

    # ---------------------------
    # Paths (kept as in your script; no change to item 4)
    # ---------------------------
    BASE_PATH = os.getenv("BASE_PATH", "")
    DATA_DIR =  os.getenv("DATA_DIR", "")
    SAVE_DIR =  os.getenv("SAVE_DIR", "")
    # DATA_DIR = os.path.join(BASE_PATH, data_dir)
    # SAVE_DIR = os.path.join(BASE_PATH, save_dir)

    # ---------------------------
    # Collect file paths and load structured elements
    # ---------------------------
    paths: list[str] = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            paths.append(os.path.join(root, file))

    elements = load_documents(paths)

    # ---------------------------
    # Assign document IDs and normalize metadata
    # ---------------------------
    doc_ids: dict[str, int] = {}
    for el in elements:
        md = el.metadata
        md_dict = md.to_dict() if hasattr(md, "to_dict") else dict(md or {})
        file_name = md_dict.get("filename") or md_dict.get("file_name") or ""
        if file_name not in doc_ids:
            doc_ids[file_name] = len(doc_ids)
        md_dict["file_name"] = file_name
        md_dict["doc_id"] = doc_ids[file_name]
        # store back as a plain dict; later we also flatten when constructing nodes
        el.metadata = md_dict

    original_document_content = "".join(getattr(el, "text", "") for el in elements)

    # ---------------------------
    # Tokenizer and truncation
    # ---------------------------
    encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate_tokens(text: str, max_tokens: int) -> str:
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])

    # Limit size of the document context
    original_document_content = _truncate_tokens(
        original_document_content,
        max_document_tokens,
    )

    # ---------------------------
    # Chunk documents into nodes
    # ---------------------------
    CHUNK_SIZE = chunk_size
    chunks = chunk_elements(
        elements,
        target_tokens=CHUNK_SIZE,
        max_tokens=max(1200, CHUNK_SIZE * 2),
    )

    # Create nodes with FLATTENED metadata to satisfy vector store constraints
    nodes = [
        TextNode(
            text=c["text"],
            metadata=_flat(c.get("metadata", {}))
        )
        for c in chunks
    ]

    # ---------------------------
    # Contextualization prompt (collapsed to a single string)
    # ---------------------------
    template = (
        "<document>{WHOLE_DOCUMENT}</document> "
        "Here is the chunk we want to situate within the whole document "
        "<chunk>{CHUNK_CONTENT}</chunk> "
        "Please give a short succinct context to situate this chunk within the overall "
        "document for the purposes of improving search retrieval of the chunk. "
        "Answer only with the succinct context and nothing else."
    )

    # ---------------------------
    # Contextual Retrieval: add succinct context before each chunk
    # ---------------------------
    idx = 0
    for node in nodes:
        content_body = node.text

        metadata = dict(node.metadata or {})
        metadata["raw_chunk"] = content_body

        chunk_tokens = encoding.encode(content_body)
        allowed_doc_tokens = max(
            0, min(max_document_tokens, context_window - len(chunk_tokens))
        )
        truncated_document = _truncate_tokens(
            original_document_content,
            allowed_doc_tokens,
        )

        prompt = template.format(
            WHOLE_DOCUMENT=truncated_document,
            CHUNK_CONTENT=content_body,
        )

        response_text = chat_completion(prompt)
        contextual_text = response_text + content_body
        nodes[idx].text = contextual_text

        # Keep these keys flat/primitive
        metadata["file_name"] = metadata.get("file_name") or ""
        metadata["section"] = idx
        metadata["doc_id"] = metadata.get("doc_id")

        anchor = (
            metadata.get("page_number")
            or metadata.get("slide_number")
            or metadata.get("slide_id")
        )
        if anchor is not None:
            metadata["anchor"] = anchor

        # Re-flatten in case anything non-scalar snuck in
        nodes[idx].metadata = _flat(metadata)

        idx += 1

        print(f'Context response from LLM => {response_text}\n For given text chunk => {content_body}')

    # ---------------------------
    # Persist indices
    # ---------------------------
    vectordb_name = db_name + "_vectordb"
    bm25db_name = db_name + "_bm25"

    # Vector DB (Chroma via LlamaIndex). save_chromadb itself should also sanitize.
    save_chromadb(
        nodes=nodes,
        save_dir=SAVE_DIR,
        db_name=vectordb_name,
        collection_name=collection_name
    )

    # BM25
    save_BM25(
        nodes=nodes,
        save_dir=SAVE_DIR,
        db_name=bm25db_name
    )


# import os
# from dotenv import load_dotenv
# import tiktoken

# from llama_index.core.schema import TextNode

# from src.openai_client import chat_completion
# from src.ingest.chunking import chunk_elements
# from src.extractors import load_documents

# from .save_vectordb import save_chromadb
# from .save_bm25 import save_BM25

# load_dotenv()

# def create_and_save_db(
#         data_dir: str,
#         collection_name: str,
#         save_dir: str,
#         db_name: str = "default",
#         chunk_size: int = 500,
#         chunk_overlap: int = 50,
#         max_document_tokens: int = 2048,
#         context_window: int = 8192,
#     ) -> None:

#     # Path directory to data storage
#     BASE_PATH = os.getenv("BASE_PATH", "")
#     DATA_DIR =  os.getenv("DATA_DIR", "")
#     SAVE_DIR =  os.getenv("SAVE_DIR", "")
#     # DATA_DIR = os.path.join(BASE_PATH, data_dir)
#     # SAVE_DIR = os.path.join(BASE_PATH, save_dir)

#     # Collect file paths and load structured elements
#     paths: list[str] = []
#     for root, _, files in os.walk(DATA_DIR):
#         for file in files:
#             paths.append(os.path.join(root, file))

#     elements = load_documents(paths)

#     # Assign document IDs and normalize metadata
#     doc_ids: dict[str, int] = {}
#     for el in elements:
#         # md = getattr(el, "metadata", {}) or {}
#         # file_name = md.get("filename") or md.get("file_name") or ""
#         # src/contextual_retrieval/save_contextual_retrieval.py  (around line 46)
#         md = el.metadata  # or however you obtain metadata
#         md_dict = md.to_dict() if hasattr(md, "to_dict") else {}

#         file_name = md_dict.get("filename") or md_dict.get("file_name") or ""
#         if file_name not in doc_ids:
#             doc_ids[file_name] = len(doc_ids)
#         md_dict["file_name"] = file_name
#         md_dict["doc_id"] = doc_ids[file_name]
#         el.metadata = md_dict

#     original_document_content = "".join(getattr(el, "text", "") for el in elements)

#     # tokenizer for token-level operations
#     encoding = tiktoken.get_encoding("cl100k_base")

#     def _truncate_tokens(text: str, max_tokens: int) -> str:
#         tokens = encoding.encode(text)
#         if len(tokens) <= max_tokens:
#             return text
#         return encoding.decode(tokens[:max_tokens])

#     # limit size of the document context
#     original_document_content = _truncate_tokens(
#         original_document_content,
#         max_document_tokens,
#     )

#     # Chunk documents into nodes
#     CHUNK_SIZE = chunk_size
#     chunks = chunk_elements(
#         elements,
#         target_tokens=CHUNK_SIZE,
#         max_tokens=max(1200, CHUNK_SIZE * 2),
#     )
#     nodes = [TextNode(text=c["text"], metadata=c.get("metadata", {})) for c in chunks]

#     # Template referred from Anthropic Blog Post
#     template = """
#             <document> 
#             {WHOLE_DOCUMENT} 
#             </document> 
#             Here is the chunk we want to situate within the whole document 
#             <chunk> 
#             {CHUNK_CONTENT} 
#             </chunk> 
#             Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
#             Answer only with the succinct context and nothing else. 
#             """

#     # Contextual Retrival : Providing context to existing Nodes [text chunks]
#     idx = 0
#     for node in nodes:
#         content_body = node.text

#         metadata = node.metadata or {}
#         metadata["raw_chunk"] = content_body

#         chunk_tokens = encoding.encode(content_body)
#         allowed_doc_tokens = max(
#             0, min(max_document_tokens, context_window - len(chunk_tokens))
#         )
#         truncated_document = _truncate_tokens(
#             original_document_content,
#             allowed_doc_tokens,
#         )

#         prompt = template.format(
#             WHOLE_DOCUMENT=truncated_document,
#             CHUNK_CONTENT=content_body,
#         )
        
#         response_text = chat_completion(prompt)
#         contextual_text = response_text + content_body
#         nodes[idx].text = contextual_text

#         metadata["file_name"] = metadata.get("file_name") or ""
#         metadata["section"] = idx
#         metadata["doc_id"] = metadata.get("doc_id")
#         anchor = (
#             metadata.get("page_number")
#             or metadata.get("slide_number")
#             or metadata.get("slide_id")
#         )
#         if anchor is not None:
#             metadata["anchor"] = anchor
#         nodes[idx].metadata = metadata

#         idx += 1

#         print(f'Context response from LLM => {response_text}\n For given text chunk => {content_body}')
    
#     vectordb_name = db_name + "_vectordb"
#     bm25db_name = db_name + "_bm25"
    
#     # Saving the Vector Database and BM25 Database
#     save_chromadb(nodes=nodes,
#                   save_dir=SAVE_DIR,
#                   db_name=vectordb_name,
#                   collection_name=collection_name)
    
#     save_BM25(nodes=nodes,
#               save_dir=SAVE_DIR,
#               db_name=bm25db_name)
