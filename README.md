# ChatSkLearn

A Scikit-learn expert chatbot built with LangGraph, LangChain, and multiple LLM providers.

## Overview

ChatSkLearn is an intelligent assistant designed to help users with Scikit-learn related questions. It uses a combination of:

- LangGraph for orchestrating the conversation flow
- LangChain for document retrieval and LLM integration
- Multiple LLM providers (Gemini, OpenAI, Anthropic, Cohere) with easy switching
- Multiple embedding providers (Gemini, OpenAI, Cohere)
- Chroma for vector storage

## Features

- Support for multiple LLM and embedding providers (defaults to Gemini)
- Query classification to determine if questions are:
  - Scikit-learn related (requiring research)
  - General questions (politely declining non-relevant queries)
  - Requiring more information
- Automated research process that:
  1. Generates a research plan
  2. Creates diverse search queries
  3. Retrieves relevant documents
  4. Synthesizes information into a helpful response
- Code-aware response formatting that preserves code blocks

## Supported LLM Providers

- **Gemini** (Google) - Default
- **OpenAI** (GPT models)
- **Anthropic** (Claude models)
- **Cohere** (Command models)

## Supported Embedding Providers

- **Gemini** (Google) - Default
- **OpenAI** (text-embedding models)
- **Cohere** (embed models)

## Project Structure

```
chatSkLearn/
├── README.md
├── .env.example
├── requirements.txt
├── config/
│   └── settings.py              # Configuration variables and  settings
├── app/
│   ├── __init__.py              # App initialization
│   ├── main.py                  # Entry point
│   ├── api/                     # API endpoints (Flask placeholder)
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Multi-provider embedding functionality
│   │   ├── llm.py               # Multi-provider LLM setup
│   │   └── utils.py             # Utility functions
│   ├── graphs/                  # LangGraph components
│   │   ├── __init__.py
│   │   ├── base.py              # Common graph components
│   │   ├── router.py            # Query analysis and routing logic
│   │   ├── researcher.py        # Research graph logic
│   │   ├── states.py            # State definitions
│   │   └── prompts.py           # System prompts
│   └── data/
│       ├── __init__.py
│       └── retriever.py         # Document retrieval functionality
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_router.py
│   ├── test_researcher.py
│   └── test_retriever.py
└── DATA/                        # Data storage (could be gitignored)
    └── chroma_store/            # Chroma vector store
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys (see `.env.example`)
6. Run the application: `python app/main.py`

## API Usage

The application exposes a simple Flask API with the following endpoints:

### `/chat` (POST)

Process chat messages with optional provider selection.

Example:

```python
import requests

response = requests.post("http://localhost:5000/chat", json={
    "messages": [
        {"role": "human", "content": "How do I implement a random forest classifier in scikit-learn?"}
    ],
    "llm_provider": "gemini",  # Optional: gemini, openai, anthropic, cohere
    "embedding_provider": "gemini"  # Optional: gemini, openai, cohere
})

print(response.json())
```

### `/providers` (GET)

Get information about available providers and current settings.

Example:

```python
import requests

response = requests.get("http://localhost:5000/providers")
print(response.json())
```

## Switching Providers

You can switch LLM and embedding providers in three ways:

1. **Environment Variables**: Set `LLM_PROVIDER` and `EMBEDDING_PROVIDER` in your .env file
2. **API Parameters**: Send provider preferences with each request
3. **Programmatically**: When using the library directly

Example of programmatic usage:

```python
from app.main import process_message

async def example():
    result = await process_message(
        messages=[{"role": "human", "content": "How to use GridSearchCV in scikit-learn?"}],
        llm_provider="openai",
        embedding_provider="openai"
    )
    print(result)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.