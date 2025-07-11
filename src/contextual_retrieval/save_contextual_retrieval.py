from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from src.azure_client import chat_completion
from .save_vectordb import save_chromadb
from .save_bm25 import save_BM25
import os
from dotenv import load_dotenv
load_dotenv()

def create_and_save_db(
        data_dir: str,
        collection_name : str,
        save_dir: str,
        db_name: str = "default",
        chunk_size: int = 500,
        chunk_overlap: int = 50
        ) -> None:

    # Path directory to data storage
    BASE_PATH = os.getenv("BASE_PATH", "")
    DATA_DIR = os.path.join(BASE_PATH, data_dir)
    SAVE_DIR = os.path.join(BASE_PATH, save_dir)

    # Hyperparameters for text splitting
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap

    # Using Azure OpenAI for contextual retrieval
    
    # Reading documents
    reader = SimpleDirectoryReader(input_dir=DATA_DIR)
    documents = reader.load_data()

    original_document_content = ""
    for page in documents:
        original_document_content += page.text

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

        prompt = template.format(WHOLE_DOCUMENT=original_document_content, 
                                 CHUNK_CONTENT=content_body)
        
        response_text = chat_completion(prompt)
        contextual_text = response_text + content_body
        nodes[idx].text = contextual_text

        metadata = node.metadata or {}
        metadata["file_name"] = metadata.get("file_name", "")
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
