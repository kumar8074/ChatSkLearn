# ===================================================================================
# Project: ChatSkLearn
# File: app/graphs/router.py
# Description: This file contains the implementation of Researcher main SkLearnAssitant Graph.
#              It also includes the Researcher sub-Graph.
# Author: LALAN KUMAR
# Created: [18-04-2025]
# Updated: [29-04-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import asyncio
import os
import sys
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, cast, Literal
from langchain_core.runnables import RunnableConfig

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from app.graphs.states import AgentState, InputState, Router
from app.graphs.prompts import (
    ROUTER_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    MORE_INFO_SYSTEM_PROMPT,
    RESEARCH_PLAN_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_PROMPT
)
from app.core.llm import get_llm
from langchain_core.messages import BaseMessage, HumanMessage
from app.graphs.researcher import create_researcher_graph
from app.core.utils import format_docs
from typing import Any

# QueryAnalyzer
async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and determine the appropriate routing"""
    llm = get_llm()
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    #response = cast(Router, await llm.with_structured_output({"logic": str, "type": str}).ainvoke(messages))
    model= llm.with_structured_output(Router) # Important for Gemini or Antropic models
    response= await model.ainvoke(messages)
    return {"router": response}

# QueryRouter
def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next steps based on query classification"""
    _type = state.router["type"]
    if _type == "scikit-learn":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type: {_type}")

# ask_for_more_info
async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information"""
    llm = get_llm()
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

# Respond to general query
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to scikit-learn."""
    llm = get_llm()
    system_prompt = GENERAL_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

# Create research plan
async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a Scikit-learn related query."""

    class Plan(TypedDict):
        """Generate research plan."""
        steps: list[str]

    llm = get_llm()
    model = llm.with_structured_output(Plan)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}

# Conduct Research
async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan."""
    researcher_graph = create_researcher_graph()
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    
    return {"documents": result["documents"], "steps": state.steps[1:]}

def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed."""
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"

async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research."""

    llm = get_llm()
    context = format_docs(state.documents)
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": prompt + "\n\nIMPORTANT: Always preserve code blocks with ```python and ``` markers. Never modify code content."}
    ] + state.messages
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

def create_router_graph():
    """Create and return the main agent graph."""
    builder = StateGraph(AgentState, input=InputState)

    builder.add_node("analyze_and_route_query", analyze_and_route_query)
    builder.add_node("ask_for_more_info", ask_for_more_info)
    builder.add_node("respond_to_general_query", respond_to_general_query)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("create_research_plan", create_research_plan)
    builder.add_node("respond", respond)

    builder.add_edge(START, "analyze_and_route_query")
    builder.add_conditional_edges("analyze_and_route_query", route_query)
    builder.add_edge("create_research_plan", "conduct_research")
    builder.add_conditional_edges("conduct_research", check_finished)
    builder.add_edge("ask_for_more_info", END)
    builder.add_edge("respond_to_general_query", END)
    builder.add_edge("respond", END)

    # Compile into a graph object that you can invoke and deploy
    graph = builder.compile()
    graph.name = "SkLearnAssistantGraph"
    
    return graph


# Example usage:
#graph=create_router_graph()
#print("Graph compiled successfully.")
#print(graph.nodes)

#input_state = AgentState(
    #messages=[HumanMessage(content="How apply PCA on images?")]
#)


#result = asyncio.run(graph.ainvoke(input_state))
#print(result)

#print("------------------------------------------------------------------------------------------",sep="\n")

#final_response = result["messages"][-1].content
#print(final_response)