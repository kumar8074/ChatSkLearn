# app/api/routes.py
"""API routes for the ChatSkLearn application."""
from flask import Flask, request, jsonify
from app.main import process_message
from config import settings
import asyncio

# This is a placeholder for the Flask application
# You would replace this with your actual Flask setup
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
async def chat():
    """Process chat messages.
    
    Request body should be a JSON with:
    - messages: List of message dictionaries (role, content)
    - llm_provider: (Optional) LLM provider to use
    - embedding_provider: (Optional) Embedding provider to use
    """
    data = request.json
    messages = data.get("messages", [])
    llm_provider = data.get("llm_provider")
    embedding_provider = data.get("embedding_provider")
    
    # Process messages through the graph
    result = await process_message(
        messages,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider
    )
    
    return jsonify({
        "response": result,
        "llm_provider": llm_provider or settings.LLM_PROVIDER,
        "embedding_provider": embedding_provider or settings.EMBEDDING_PROVIDER
    })

@app.route("/providers", methods=["GET"])
def get_providers():
    """Get available providers and current settings."""
    return jsonify({
        "available_llm_providers": [
            settings.LLM_PROVIDER_GEMINI,
            settings.LLM_PROVIDER_OPENAI,
            settings.LLM_PROVIDER_ANTHROPIC,
            settings.LLM_PROVIDER_COHERE
        ],
        "available_embedding_providers": [
            settings.EMBEDDING_PROVIDER_GEMINI,
            settings.EMBEDDING_PROVIDER_OPENAI,
            settings.EMBEDDING_PROVIDER_COHERE
        ],
        "current_llm_provider": settings.LLM_PROVIDER,
        "current_embedding_provider": settings.EMBEDDING_PROVIDER
    })

if __name__ == "__main__":
    app.run(debug=True)