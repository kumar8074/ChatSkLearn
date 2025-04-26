# ===================================================================================
# Project: ChatSkLearn
# File: app/retriever/retriever.py
# Description: This file Loads the persisted vectorDB and returns a retriever instance.
#              Defaults to ChromaDB Instance.
# Author: LALAN KUMAR
# Created: [15-04-2025]
# Updated: [26-04-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

# Dynamically add the project root directory to sys.path
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
    
    Args:
        persist_directory: Directory to persist the vector store. If None, uses the default from settings.
        embedding_provider: The embedding provider to use. If None, uses the default from settings.
        
    Returns:
        A vector store instance.
    """
    persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
    embeddings = get_embeddings(embedding_provider)
    
    # Load the persisted directory
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
        persist_directory: Directory to persist the vector store. If None, uses the default from settings.
        embedding_provider: The embedding provider to use. If None, uses the default from settings.
        search_kwargs: Additional search parameters to pass to the retriever.
        
    Returns:
        A retriever instance.
    """
    search_kwargs = search_kwargs or {"k": 5}
    vector_store = get_vector_store(persist_directory, embedding_provider)
    
    #print(vector_store._collection.count()) # If 0, VectorStore is empty
    
    # Convert it into a retriever
    return vector_store.as_retriever(search_kwargs=search_kwargs)

#get_retriever()