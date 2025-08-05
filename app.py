from src.tools.rag_workflow import RAGWorkflow
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
from src.logging_config import get_logger
from src.contextual_retrieval.save_contextual_retrieval import create_and_save_db
from typing import List

load_dotenv()

logger = get_logger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserQuery(BaseModel):
    query: str


w = RAGWorkflow()
w._timeout = 120.0


async def RAG_chat(w, query):
    retriever = await w.run(collection_name=os.getenv("COLLECTION_NAME"))
    result = await w.run(query=query, retriever=retriever)
    nodes = result["nodes"]
    answer = result["answer"]

    sources: List[dict] = []
    for n in nodes:
        file_path = n.node.metadata.get("file_path")
        link = (
            Path(file_path).resolve().as_uri()
            if file_path
            else None
        )
        sources.append(
            {
                "file": n.node.metadata.get("file_name"),
                "path": file_path,
                "text": n.node.metadata.get("raw_chunk"),
                "link": link,
            }
        )
    filenames = [os.path.basename(n.node.metadata.get("file_name", "")) for n in nodes if n.node.metadata.get("file_name")]
    if filenames:
        logger.info("Documents accessed: %s", ", ".join(filenames))

    response_obj = await answer.get_response()
    final_answer = response_obj.response

    return {"answer": final_answer, "sources": sources}


@app.post("/rag-chat")
async def root(user_query: UserQuery):
    logger.info(f"User query: {user_query.query}")
    try:
        return await RAG_chat(w=w, query=user_query.query)
    except Exception:
        logger.exception("Error processing /rag-chat request")
        raise


# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     """Upload a document and rebuild the vector store."""
#     data_dir = os.path.join(os.getenv("BASE_PATH", ""), os.getenv("DATA_DIR", ""))
#     os.makedirs(data_dir, exist_ok=True)
#     file_path = os.path.join(data_dir, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
#     logger.info("Saved uploaded file: %s", file_path)
#     try:
#         create_and_save_db(
#             data_dir=os.getenv("DATA_DIR", ""),
#             save_dir=os.getenv("SAVE_DIR", ""),
#             collection_name=os.getenv("COLLECTION_NAME"),
#             db_name="cook_book_db",
#         )
#     except Exception:
#         logger.exception("Error rebuilding database")
#         raise
#     return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
