from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.readers.file.pymu_pdf import PyMuPDFReader
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()

from src.openai_client import chat_completion

from .save_vectordb import save_chromadb
from .save_bm25 import save_BM25

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

    # Path directory to data storage
    BASE_PATH = os.getenv("BASE_PATH", "")
    DATA_DIR =  os.getenv("DATA_DIR", "")
    SAVE_DIR =  os.getenv("SAVE_DIR", "")
    # DATA_DIR = os.path.join(BASE_PATH, data_dir)
    # SAVE_DIR = os.path.join(BASE_PATH, save_dir)

    # Hyperparameters for text splitting
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap

    # Reading documents
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        file_extractor={".pdf": PyMuPDFReader()},
        recursive=True,
    )
    documents = reader.load_data()

    for doc in documents:
        metadata = doc.metadata or {}
        file_path = metadata.get("file_path")
        if file_path:
            rel = os.path.relpath(file_path, DATA_DIR)
            metadata["file_path"] = rel
        doc.metadata = metadata

    original_document_content = ""
    for page in documents:
        original_document_content += page.text

    # tokenizer for token-level operations
    encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate_tokens(text: str, max_tokens: int) -> str:
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])

    # limit size of the document context
    original_document_content = _truncate_tokens(
        original_document_content,
        max_document_tokens,
    )

    # Initializing text splitter
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" ",
    )

    # Splitting documents to Nodes [text chunks]
    nodes = splitter.get_nodes_from_documents(documents)

    # Template referred from Anthropic Blog Post
    template = """
            <document> 
            {WHOLE_DOCUMENT} 
            </document> 
            Here is the chunk we want to situate within the whole document 
            <chunk> 
            {CHUNK_CONTENT} 
            </chunk> 
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
            Answer only with the succinct context and nothing else. 
            """

    # Contextual Retrival : Providing context to existing Nodes [text chunks]
    idx = 0
    for node in nodes:
        content_body = node.text

        metadata = node.metadata or {}
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

        metadata["file_name"] = metadata.get("file_name") or metadata.get("file_path") or ""
        metadata["section"] = idx
        nodes[idx].metadata = metadata

        idx += 1

        print(f'Context response from LLM => {response_text}\n For given text chunk => {content_body}')
    
    vectordb_name = db_name + "_vectordb"
    bm25db_name = db_name + "_bm25"
    
    # Saving the Vector Database and BM25 Database
    save_chromadb(nodes=nodes,
                  save_dir=SAVE_DIR,
                  db_name=vectordb_name,
                  collection_name=collection_name)
    
    save_BM25(nodes=nodes,
              save_dir=SAVE_DIR,
              db_name=bm25db_name)
