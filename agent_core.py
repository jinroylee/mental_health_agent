"""
Tiny wrapper so UI / CLI only calls agent().
"""

import mlflow
from graphs.mh_graph import build_graph

mlflow.set_experiment("mh-agent-dev")
mlflow.langchain.autolog()

# Create the graph instance
_graph = build_graph()

def agent(user_input: str, user_id: str = "default_user", user_locale: str = "US", await_feedback_prev: bool = False) -> str:
    """Run the mental health assistant."""
    with mlflow.start_run(nested=True):
        # Format input state to match ChatState structure
        state_in = {
            "last_user_msg": user_input,
            "user_id": user_id,
            "user_locale": user_locale,
            "chat_history": [],         # Initialize empty chat history
            "awaiting_feedback": await_feedback_prev,
        }

        # Invoke the graph
        state_out = _graph.invoke(state_in)

        # Extract the assistant's response from chat_history
        if state_out.get("chat_history"):
            # Get the last message from chat history
            last_message = state_out["chat_history"][-1]
            print(f"await_feedback_prev: {await_feedback_prev}")
            await_feedback_new = state_out.get("awaiting_feedback", False)
            print(f"await_feedback_new: {await_feedback_new}")
            return last_message.content, await_feedback_new 
        else:
            return "I'm sorry, I couldn't process your message right now."
