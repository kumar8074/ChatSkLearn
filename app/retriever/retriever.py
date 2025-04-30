# ===================================================================================
# Project: ChatSkLearn
# File: app/retriever/retriever.py
# Description: This file Loads the persisted vectorDB and returns a retriever instance.
#              Defaults to ChromaDB Instance.
# Author: LALAN KUMAR
# Created: [15-04-2025]
# Updated: [30-04-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings
from app.core.embeddings import get_embeddings

def get_vector_store(
    persist_directory: Optional[str] = None,
    embedding_provider: Optional[str] = None
) -> VectorStore:
    """Get a vector store instance with the specified embeddings.

    If persist_directory is not provided, it is dynamically selected based on the LLM provider.

    Args:
        persist_directory: Directory to persist the vector store.
        embedding_provider: The embedding provider to use. If None, uses the default from settings.

    Returns:
        A vector store instance.
    """
    # Dynamically determine directory based on LLM provider
    if persist_directory is None:
        llm_provider = settings.LLM_PROVIDER.lower()
        persist_directory = f"DATA/chroma_store_{llm_provider}"

    embeddings = get_embeddings(embedding_provider)
    #print(f"vectorStore loaded from: {persist_directory} using embeddings: {embeddings}")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def get_retriever(
    persist_directory: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    search_kwargs: Optional[dict] = None
):
    """Initialize and return a retriever.

    Args:
        persist_directory: Directory to persist the vector store. Dynamically derived if not provided.
        embedding_provider: The embedding provider to use. If None, uses the default from settings.
        search_kwargs: Additional search parameters to pass to the retriever.

    Returns:
        A retriever instance.
    """
    search_kwargs = search_kwargs or {"k": 5}
    vector_store = get_vector_store(persist_directory, embedding_provider)
    
    #print(vector_store._collection.count()) # If 0, VectorStore is empty
    
    return vector_store.as_retriever(search_kwargs=search_kwargs)

#get_retriever()