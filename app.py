from src.tools.rag_workflow import RAGWorkflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from src.logging_config import get_logger
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

    sources: List[dict] = [
        {"file": n.node.metadata.get("file_name"), "text": n.node.metadata.get("raw_chunk")}
        for n in nodes
    ]
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
