# ===================================================================================
# Project: ChatSkLearn
# File: tools/retriever_tool.py
# Description: This modeule contains the code to create a retriever tool using the Chroma database.
# Author: LALAN KUMAR
# Created: [21-04-2025]
# Updated: [21-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

from langchain.tools.retriever import create_retriever_tool
from loaders.chroma_loader import load_chroma

def build_retrieval_tool():
    """
    Build a retrieval tool using the Chroma database.
    """
    # Load the Chroma database
    vectordb = load_chroma()

    # Create a retriever from the database
    retriever = vectordb.as_retriever()

    # Create a retriever tool
    retrieval_tool = create_retriever_tool(
        retriever,
        name="sklearn_doc_retriever",
        description="Use this tool to search and return relevant information from the scikit-learn documentation. Use it only when the question is about scikit-learn functionality, API usage, guides, examples, or references. Do NOT use it for greetings or unrelated general queries."
    )

    retrieval_tool.metadata= {"source": "scikit-learn-docs"}
    
    return retrieval_tool