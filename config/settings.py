# ===================================================================================
# Project: ChatSkLearn
# File: config/settings.py
# Description: This file contains configuration variables and settings for the ChatSkLearn project
# Author: LALAN KUMAR
# Created: [15-04-2025]
# Updated: [26-04-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# Supported LLM models
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_ANTHROPIC = "anthropic"
LLM_PROVIDER_COHERE = "cohere"

# Supported embedding models
EMBEDDING_PROVIDER_GEMINI = "gemini"
EMBEDDING_PROVIDER_OPENAI = "openai"
EMBEDDING_PROVIDER_COHERE = "cohere"

# Default providers
DEFAULT_LLM_PROVIDER = LLM_PROVIDER_GEMINI
DEFAULT_EMBEDDING_PROVIDER = EMBEDDING_PROVIDER_GEMINI

# Get provider from environment or use default
LLM_PROVIDER = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)

# API Keys
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Set environment variables for libraries
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Model settings - Gemini
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

# Model settings - OpenAI
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-2025-04-14")

# Model settings - Anthropic
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-opus-4-20250514")

# Model settings - Cohere
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
COHERE_LLM_MODEL = os.getenv("COHERE_LLM_MODEL", "command-a-03-2025")

# Database settings
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "DATA/chroma_store")

#print("Settings loaded successfully.")