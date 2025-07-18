"""
LangGraph definition for the mental-health agent.
"""

from __future__ import annotations
from typing import TypedDict, List, Literal
import os, dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.moderation import OpenAIModerationChain
from langchain_community.vectorstores import Pinecone as PineconeStore
import pinecone
from pathlib import Path

# ---------- Environment ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

# ---------- State ----------
class ChatState(TypedDict, total=False):
    user_input: str
    context: List[str]
    answer: str
    safe: Literal["ok", "crisis"]

# ---------- Shared resources ----------
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
retriever = PineconeStore.from_existing_index(
    "mh-agent",
    OpenAIEmbeddings(model="text-embedding-ada-002")
).as_retriever(search_kwargs={"k": 4})

moderation = OpenAIModerationChain()
memory     = ConversationBufferMemory(return_messages=True)

SYSTEM_PROMPT = (
    "You are a supportive mental-health assistant.\n"
    "Use the provided context to answer with empathy and practical advice.\n"
    "If the user requests medical diagnosis or expresses intent to self-harm, "
    "provide crisis resources and encourage professional help."
)

# ---------- Nodes ----------
def node_moderate(state: ChatState) -> ChatState:
    """
    Raises an exception if content is disallowed; otherwise marks safe=ok.
    """
    moderation.check(state["user_input"])
    return {**state, "safe": "ok"}

def node_retrieve(state: ChatState) -> ChatState:
    docs  = retriever.invoke(state["user_input"])
    texts = [d.page_content for d in docs]
    return {**state, "context": texts}

def node_generate(state: ChatState) -> ChatState:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{''.join(state['context'])}\n\n"
        f"User:\n{state['user_input']}"
    )
    answer = llm.invoke(prompt).content
    memory.chat_memory.add_user_message(state["user_input"])
    memory.chat_memory.add_ai_message(answer)
    return {**state, "answer": answer}

def node_crisis(_: ChatState) -> ChatState:
    msg = (
        "I'm really sorry you're feeling this way. "
        "If you think you might harm yourself or are in danger, "
        "please call your local emergency number (e.g., 911 in the U.S.) "
        "or the Suicide & Crisis Lifeline (988 in the U.S.). "
        "You can also find help lines worldwide at https://findahelpline.com."
    )
    return {"answer": msg}

# ---------- Graph ----------
def build_graph():
    sg = StateGraph(ChatState)

    sg.add_node("moderate",  node_moderate)
    sg.add_node("retrieve",  node_retrieve)
    sg.add_node("generate",  node_generate)
    sg.add_node("crisis",    node_crisis)

    # branching after moderation
    def route(state: ChatState):
        return "crisis" if state.get("safe") != "ok" else "retrieve"

    sg.add_conditional_edges(
        "moderate", route,
        {"retrieve": "retrieve", "crisis": "crisis"}
    )

    sg.add_edge("retrieve", "generate")
    sg.add_edge("crisis",   END)
    sg.add_edge("generate", END)

    return sg.compile()

graph_executor = build_graph()
