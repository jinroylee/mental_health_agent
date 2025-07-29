import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_core import agent

st.set_page_config(page_title="Mental-Health Support Chat", page_icon="🧘")

# Sidebar for user settings
with st.sidebar:
    st.header("⚙️ User Settings")
    
    # Initialize session state for user settings if not exists
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "default_user"
    if "user_locale" not in st.session_state:
        st.session_state["user_locale"] = "US"
    
    # User ID input
    user_id = st.text_input(
        "User ID:",
        value=st.session_state["user_id"],
        help="Enter a unique identifier for yourself"
    )
    
    # Locale selection
    locale_options = ["US", "UK", "CA", "AU", "DE", "FR", "ES", "IT", "JP", "Other"]
    user_locale = st.selectbox(
        "Locale:",
        options=locale_options,
        index=locale_options.index(st.session_state["user_locale"]) if st.session_state["user_locale"] in locale_options else 0,
        help="Select your region/locale"
    )
    
    # Update session state when values change
    st.session_state["user_id"] = user_id
    st.session_state["user_locale"] = user_locale
    
    # Display current settings
    st.markdown("---")
    st.markdown("**Current Settings:**")
    st.markdown(f"• User ID: `{user_id}`")
    st.markdown(f"• Locale: `{user_locale}`")

st.title("🧘 Mental-Health Support Chat")
st.markdown(
    """
    *I'm an AI assistant and **not** a licensed clinician.*
    If you feel unsafe with your thoughts, please seek professional help immediately.
    """
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize awaiting_feedback state if not exists
if "awaiting_feedback" not in st.session_state:
    st.session_state["awaiting_feedback"] = False

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User input
user_msg = st.chat_input("How are you feeling today?")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.spinner("Thinking…"):
        # Use the stored awaiting_feedback state from previous interaction
        answer, await_feedback_new = agent(
            user_msg, 
            st.session_state["user_id"], 
            st.session_state["user_locale"], 
            await_feedback_prev=st.session_state["awaiting_feedback"]
        )
        # Update the session state with the new awaiting_feedback value
        st.session_state["awaiting_feedback"] = await_feedback_new

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
