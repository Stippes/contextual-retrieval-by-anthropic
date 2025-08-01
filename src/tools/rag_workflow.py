from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.core import PromptTemplate
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from src.db.read_db import SemanticBM25Retriever
from src.openai_client import OpenAIChatClient
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.callbacks import CallbackManager
from typing import Any, Optional
from pydantic import Field
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]

# prompt template
template = (
    "The below provided is the context from a bunch of cook books \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question like a cheif in bullet points and elaborate: {query_str}\n"
)

qa_template = PromptTemplate(template)


class OpenAIChatLLM(CustomLLM):
    # declare `client` as a field so Pydantic knows about it
    client: OpenAIChatClient = Field(default_factory=OpenAIChatClient)

    def __init__(self, client: Optional[OpenAIChatClient] = None) -> None:
        # if you want to override the default, object.__setattr__ to avoid Pydantic checks:
        super().__init__(callback_manager=CallbackManager([]))
        if client is not None:
            object.__setattr__(self, "client", client)

    def __init__(self, client: OpenAIChatClient | None = None) -> None:
        super().__init__(callback_manager=CallbackManager([]))
        self.client = client or OpenAIChatClient()

    @classmethod
    def class_name(cls) -> str:
        return "openai_chat_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=-1,
            is_chat_model=True,
            model_name=self.client.model,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text = self.client.chat([{"role": "user", "content": prompt}])
        return CompletionResponse(text=text)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = self.client.chat([{"role": "user", "content": prompt}])

        def gen() -> CompletionResponseGen:
            yield CompletionResponse(text=text, delta=text)

        return gen()

# RAG using workflow
class RAGWorkflow(Workflow):
    
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:

        collection_name = ev.get("collection_name")
        if not collection_name:
            return None

        retriever = SemanticBM25Retriever(collection_name=collection_name)

        return StopEvent(result=retriever)

    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        
        query = ev.get("query")
        retriever = ev.get("retriever")

        if not query:
            return None

        print(f"Query the database with: {query}")

        await ctx.set("query", query)

        if retriever is None:
            print("Index is empty, load some documents before querying!")
            return None

        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")

        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:

        llm = OpenAIChatLLM()
        summarizer = CompactAndRefine(
            llm=llm,
            streaming=True,
            verbose=True,
            text_qa_template=qa_template,
        )
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)

        return StopEvent(result={"answer": response, "nodes": ev.nodes})
    
