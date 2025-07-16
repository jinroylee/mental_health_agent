import streamlit as st
from agent_core import agent

st.set_page_config(page_title="Mental-Health Support Chat", page_icon="🧘")
st.title("🧘 Mental-Health Support Chat")
st.markdown(
    """
    *I’m an AI assistant and **not** a licensed clinician.*
    If you feel unsafe with your thoughts, please seek professional help immediately.
    """
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
        answer = agent(user_msg)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
