# ===================================================================================
# Project: ChatSkLearn
# File: app/processor/data_ingestion.py
# Description: This file Loads the data from URLs and store it in a VectorStoreDB (Defaults to CHROMA)
# Author: LALAN KUMAR
# Created: [15-04-2025]
# Updated: [30-04-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================
# CAUTION: DO NOT RUN this file until you want to make changes to the vectorStoreDB.
# RECOMMENDATION: It is recommended to run this file multiple times and persist the vectorDB using all the embedding providers.
#                 Currently it persists the vectorDB locally which is efiicient for ChatSklearn application types.
#                 However you can modify the script and persist the vectorDB externally to third party databases.

import os
import sys

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from logger import logging
from config import settings
from app.core.embeddings import get_embeddings

def load_urls(file_path: str) -> list[str]:
    """Load and clean URLs from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def load_documents(urls: list[str]):
    """Load documents from web pages."""
    loader = WebBaseLoader(web_paths=urls)
    return loader.load()

def split_documents(documents):
    """Split large documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def persist_vector_db(docs, embeddings, persist_directory):
    """Save documents to Chroma vector store."""
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info(f"VectorDB persisted at: {persist_directory}")

def main():
    urls_file_path = "processor/successful_urls.txt"
    logging.info(f"Loading URLs from {urls_file_path}")
    urls = load_urls(urls_file_path)

    logging.info(f"Loaded {len(urls)} URLs...")
    
    logging.info(f"Loading Documents from URLs....(This might take a while have a coffe break)")
    documents = load_documents(urls)
    logging.info(f"Loaded {len(documents)} documents...")
    
    logging.info(f"Splitting documents into chunks...")
    docs = split_documents(documents)
    logging.info(f"Split into {len(docs)} chunks...")

    # Get dynamic embedding model
    embedding_provider = settings.EMBEDDING_PROVIDER
    embeddings = get_embeddings(embedding_provider)
    logging.info(f"Using Embedding provider: {embeddings}")

    # Set output directory based on provider name
    persist_directory = f"DATA/chroma_store_{embedding_provider.lower()}"
    os.makedirs(persist_directory, exist_ok=True)
    
    logging.info(f"Persisting vectorDB...(This might take a while)")
    persist_vector_db(docs, embeddings, persist_directory)

if __name__ == "__main__":
    main()
