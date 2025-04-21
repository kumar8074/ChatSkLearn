# ===================================================================================
# Project: ChatSkLearn
# File: graphs/researcher_graph.py
# Description: This module defines the core structure and functionality of the researcher graph,
#              which is responsible for generating search queries and parallel retrieving relevant documents.
# Author: LALAN KUMAR
# Created: [21-04-2025]
# Updated: [21-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

from typing import TypedDict, cast

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from states.states import QueryState, ResearcherState
from tools.retriever_tool import build_retrieval_tool
from initializer import get_llm
from prompts.prompts import GENERATE_QUERIES_SYSTEM_PROMPT


class Response(TypedDict):
    queries: list[str]


async def generate_queries(state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[str]]:
    """Generate multiple search queries based on the original user question."""
    model = get_llm().with_structured_output(Response)

    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "user", "content": state.question},
    ]

    response = cast(Response, await model.ainvoke(messages))
    return {"queries": response["queries"]}


async def retrieve_documents(state: QueryState, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """Use the retriever to fetch relevant documents for a given query."""
    _, retriever = build_retrieval_tool()
    response = await retriever.ainvoke(state.query)
    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Dispatch retrieval for each generated query in parallel."""
    return [Send("retrieve_documents", QueryState(query=q)) for q in state.queries]


# Build the researcher subgraph
builder = StateGraph(ResearcherState)

builder.add_node("generate_queries", generate_queries)
builder.add_node("retrieve_documents", retrieve_documents)

builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_documents"]
)
builder.add_edge("retrieve_documents", END)

# Compile into a callable subgraph
researcher_graph = builder.compile()
researcher_graph.name = "ResearcherGraph"
