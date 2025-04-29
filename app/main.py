# app/main.py
"""Entry point for the ChatSkLearn application."""
from app.graphs.router import create_router_graph
from typing import Optional, Dict, Any, List
from config import settings

def create_app(
    llm_provider: Optional[str] = None,
    embedding_provider: Optional[str] = None
):
    """Initialize the application.
    
    Args:
        llm_provider: LLM provider to use. If None, uses the default from settings.
        embedding_provider: Embedding provider to use. If None, uses the default from settings.
        
    Returns:
        Dict containing the application components.
    """
    # Store the providers in settings if provided
    if llm_provider:
        settings.LLM_PROVIDER = llm_provider
    if embedding_provider:
        settings.EMBEDDING_PROVIDER = embedding_provider
        
    # Here you would set up Flask or FastAPI
    return {
        "graph": create_router_graph(),
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER
    }

async def process_message(
    messages: List[Dict[str, str]],
    llm_provider: Optional[str] = None,
    embedding_provider: Optional[str] = None
) -> Dict[str, Any]:
    """Process user messages through the graph.
    
    Args:
        messages: List of message dictionaries (role, content).
        llm_provider: LLM provider to use. If None, uses the default from settings.
        embedding_provider: Embedding provider to use. If None, uses the default from settings.
        
    Returns:
        Dict containing the processed result.
    """
    # Store the providers in settings if provided
    if llm_provider:
        settings.LLM_PROVIDER = llm_provider
    if embedding_provider:
        settings.EMBEDDING_PROVIDER = embedding_provider
        
    graph = create_router_graph()
    return await graph.ainvoke({"messages": messages})

if __name__ == "__main__":
    import asyncio
    
    # Example usage
    async def main():
        # Using default providers (Gemini)
        result = await process_message([
            {"role": "human", "content": "How do I implement a random forest classifier in scikit-learn?"}
        ])
        print("Result with default provider (Gemini):", result)
        
        # Using OpenAI (if API key is available)
        if settings.OPENAI_API_KEY:
            result = await process_message(
                [{"role": "human", "content": "How do I implement a random forest classifier in scikit-learn?"}],
                llm_provider=settings.LLM_PROVIDER_OPENAI,
                embedding_provider=settings.EMBEDDING_PROVIDER_OPENAI
            )
            print("Result with OpenAI:", result)
    
    asyncio.run(main())