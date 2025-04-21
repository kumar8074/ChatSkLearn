# ===================================================================================
# Project: ChatSkLearn
# File: initializer.py
# Description: Initializes the LLM and Embedding models for the ChatSkLearn project.
# Author: LALAN KUMAR
# Created: [21-04-2025]
# Updated: [21-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Set the API Key
os.environ["GEMINI_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# LLM initialization
def get_llm(model: str = "gemini-2.0-flash"):
    return ChatGoogleGenerativeAI(model==model)

# Embedding initialization
def get_embeddings(model: str = "models/text-embedding-004"):
    return GoogleGenerativeAIEmbeddings(model=model)