"""
Tiny wrapper so UI / CLI only calls agent().
"""

import mlflow
from graphs.mh_graph import build_graph
from langchain_core.messages import AIMessage, HumanMessage

mlflow.set_experiment("mh-agent-dev")
mlflow.langchain.autolog()

# Create the graph instance
_graph = build_graph()

def agent(user_input: str, user_id: str = "default_user", user_locale: str = "US") -> str:
    """Run the mental health assistant."""
    with mlflow.start_run(nested=True):
        # Format input state to match ChatState structure
        state_in = {
            "last_user_msg": user_input,
            "user_id": user_id,
            "user_locale": user_locale,
            "chat_history": [],      
        }

        # Invoke the graph
        state_out = _graph.invoke(state_in)

        # Extract AI messages that come after the current user message
        chat_history = state_out.get("chat_history", [])
        if chat_history:
            # Find the last occurrence of the current user message
            user_msg_index = -1
            for i in range(len(chat_history) - 1, -1, -1):  # Search backwards
                if (isinstance(chat_history[i], HumanMessage) and 
                    chat_history[i].content == user_input):
                    user_msg_index = i
                    break
            
            # Collect AI messages that appear after the user message
            if user_msg_index >= 0:
                ai_messages = [
                    msg.content for msg in chat_history[user_msg_index + 1:]
                    if isinstance(msg, AIMessage)
                ]
            else:
                # Fallback: if we can't find the user message, collect all AI messages
                ai_messages = [msg.content for msg in chat_history if isinstance(msg, AIMessage)]
            
            if ai_messages:
                # Merge multiple AI messages into a single response with clear separators
                if len(ai_messages) == 1:
                    merged_response = ai_messages[0]
                else:
                    # Use clear separators between different response sections
                    merged_response = "\n\n---\n\n".join(ai_messages)
                
                return merged_response
            else:
                return "I'm sorry, I couldn't process your message right now.", False
        else:
            return "I'm sorry, I couldn't process your message right now.", False
