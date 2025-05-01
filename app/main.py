from flask import Flask, render_template, request, jsonify, session
import os
import sys
from uuid import uuid4
import asyncio
from langchain_core.messages import HumanMessage

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the router graph and settings
from app.graphs.router import create_router_graph
from config import settings

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', str(uuid4()))

# Create the router graph
router_graph = create_router_graph()

# Define models that are available and their corresponding LLM providers
MODELS = {
    "gpt-4o": "OpenAI GPT-4o",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "gemini-2.0-flash": "Gemini 2.0 flash",
    "command": "Cohere Command"
}

# Map frontend model selections to LLM providers
MODEL_TO_PROVIDER = {
    "gpt-4o": settings.LLM_PROVIDER_OPENAI,
    "claude-3-sonnet": settings.LLM_PROVIDER_ANTHROPIC,
    "gemini-2.0-flash": settings.LLM_PROVIDER_GEMINI,
    "command": settings.LLM_PROVIDER_COHERE
}

# Sample suggestions
SUGGESTIONS = [
    "How do I use RandomForestClassifier in scikit-learn?",
    "What's the difference between train_test_split and StratifiedKFold?",
    "How can I tune hyperparameters in scikit-learn?",
    "Explain how to handle imbalanced datasets in scikit-learn."
]

# In-memory chat history storage
# In a production app, you'd use a database instead
chat_history = {}
conversation_states = {}

@app.route('/')
def index():
    # Set default model if none is selected
    if 'selected_model' not in session:
        session['selected_model'] = 'gemini-2.0-flash'
    
    # Generate a unique session ID if none exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid4())
        chat_history[session['session_id']] = []
        conversation_states[session['session_id']] = []
    
    return render_template('index.html', 
                          models=MODELS, 
                          selected_model=session['selected_model'],
                          selected_model_name=MODELS[session['selected_model']],
                          suggestions=SUGGESTIONS,
                          chat_history=chat_history.get(session['session_id'], []))

@app.route('/select_model', methods=['POST'])
def select_model():
    model_id = request.form.get('model')
    if model_id in MODELS:
        session['selected_model'] = model_id
        
        # Update the LLM provider based on the selected model
        llm_provider = MODEL_TO_PROVIDER[model_id]
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["EMBEDDING_PROVIDER"] = llm_provider
        
        # Update settings
        settings.LLM_PROVIDER = llm_provider
        settings.EMBEDDING_PROVIDER = llm_provider
        
        return jsonify({"status": "success", "model": MODELS[model_id]})
    return jsonify({"status": "error", "message": "Invalid model selection"})

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form.get('message', '').strip()
    model_id = session.get('selected_model', 'gemini-2.0-flash')
    
    if not message:
        return jsonify({"status": "error", "message": "Message cannot be empty"})
    
    # Add user message to history
    session_id = session.get('session_id')
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    # Add the user's message to chat history
    chat_history[session_id].append(message)
    
    # Make sure we're using the right LLM provider
    llm_provider = MODEL_TO_PROVIDER[model_id]
    os.environ["LLM_PROVIDER"] = llm_provider
    os.environ["EMBEDDING_PROVIDER"] = llm_provider
    settings.LLM_PROVIDER = llm_provider
    settings.EMBEDDING_PROVIDER = llm_provider
    
    # Prepare message for the graph
    human_message = HumanMessage(content=message)
    
    # Get the conversation state or create a new one
    if session_id not in conversation_states:
        conversation_states[session_id] = []
    
    # Add the new message to the conversation state
    current_state = conversation_states[session_id]
    current_state.append(human_message)
    
    # Run the graph (asynchronously)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Invoke the graph with the current state
        result = loop.run_until_complete(
            router_graph.ainvoke({"messages": current_state})
        )
        
        # Extract the response
        response_message = result["messages"][-1]
        response_content = response_message.content
        
        # Update the conversation state with the assistant's response
        current_state.append(response_message)
        conversation_states[session_id] = current_state
        
        return jsonify({
            "status": "success",
            "message": message,
            "response": response_content,
            "model": MODELS[model_id]
        })
    finally:
        loop.close()

@app.route('/get_suggestions')
def get_suggestions():
    return jsonify({"suggestions": SUGGESTIONS})

@app.route('/get_chat_history')
def get_chat_history():
    session_id = session.get('session_id')
    return jsonify({"history": chat_history.get(session_id, [])})

@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    session_id = session.get('session_id')
    if session_id in chat_history:
        chat_history[session_id] = []
    if session_id in conversation_states:
        conversation_states[session_id] = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000,debug=True)
