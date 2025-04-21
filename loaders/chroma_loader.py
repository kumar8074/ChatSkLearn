# ===================================================================================
# Project: ChatSkLearn
# File: loaders/chroma_loader.py
# Description: This file loads the Chroma vector store from a specified directory.
# Author: LALAN KUMAR
# Created: [21-04-2025]
# Updated: [21-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================
from langchain_chroma import Chroma
from initializer import get_embeddings

def load_chroma(persist_dir: str = "DATA/chroma_store"):
    """
    Load the Chroma vector store from the specified directory.

    Args:
        persist_dir (str): Path to the directory where the Chroma vector store is persisted.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_dir, 
                  embedding_function=embeddings)