# ===================================================================================
# Project: ChatSkLearn
# File: states/states.py
# Description: This modeule defines the state schemas for the graphs
# Author: LALAN KUMAR
# Created: [21-04-2025]
# Updated: [21-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

from typing import TypedDict, List, Optional, Dict


# Common base state shared across graphs
class BaseState(TypedDict):
    request_id: str
    user_query: str
    

# For the retriever graph (single retrieval task)
class RetrieverState(BaseState):
    documents: Optional[List[Dict]]
    

# For the researcher graph
class QueryState(TypedDict):
    query: str

# For the researcher grpah (initial + post-query generation state)
class ResearcherState(BaseState):
    question: str
    queries: Optional[List[str]] 


# For the researcher graph (multi-task research + summary)
class PlannerState(BaseState):
    plan: Optional[List[Dict]]
    
    
class ParallelExecutionState(PlannerState):
    task_states: Optional[List[Dict]]
    summary_markdown: Optional[str]