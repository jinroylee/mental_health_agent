from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Literal, TypedDict

import dotenv
from langchain.chains.moderation import OpenAIModerationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langchain.schema import AIMessage, BaseMessage
# from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment & shared resources
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["INDEX_NAME"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.3)
_embed = OpenAIEmbeddings(model="text-embedding-ada-002")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
retriever = PineconeVectorStore(index=index, embedding=_embed).as_retriever(
    search_kwargs={"k": 4}
)

moderation_chain = OpenAIModerationChain()
conversation_memory = ConversationBufferMemory(return_messages=True)

CRISIS_RESOURCE_FALLBACK = (
    "If you believe you may harm yourself or others, please reach out for immediate help. In the U.S. call 988 or visit https://988lifeline.org. If you are outside the U.S., search online for a local crisis helpline in your country. You are not alone and help is available."  # noqa: E501
)

SYSTEM_PROMPT = (
    "You are a supportive mental‑health assistant.\n"
    "Use the provided context to answer with empathy and practical advice.\n"
    "If the user requests medical diagnosis or expresses intent to self‑harm, "
    "provide crisis resources and encourage professional help."
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class ChatState(TypedDict, total=False):
    """TypedDict that defines the mutable state carried through the graph."""

    # Session‑static keys (set at START)
    user_id: str
    user_locale: str

    # Per‑turn keys
    last_user_msg: str
    mood: Literal[
        "neutral",
        "anxious",
        "sad",
        "angry",
        "happy",
        "stressed",
    ]
    risk_level: Literal["safe", "crisis"]
    detected_distortion: str  # e.g. "catastrophizing" or ""
    skill_script: str  # selected coping skill instructions
    prior_summary: str  # summary retrieved at session start

    # Growing artifacts
    chat_history: List[BaseMessage]

    # Flag to terminate session
    end_session: bool


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def classify_emotion_and_risk(message: str) -> tuple[str, str]:
    """Very lightweight emotion + risk classifier.

    * Leverages OpenAI moderation for self‑harm detection.
    * Uses an LLM zero‑shot call for coarse emotion tagging.

    Returns (mood, risk_level)
    """
    print("@@@ classify_emotion_and_risk @@@")
    print(message)
    # 1. Self‑harm check via moderation
    moderation_res = moderation_chain.run(message)
    print("@@@ moderation_res @@@")
    print(moderation_res)
    if moderation_res.get("self_harm"):
        logger.info("Moderation flagged self‑harm. Escalating to crisis path.")
        return "stressed", "crisis"

    # 2. Simple LLM emotion classification (could be replaced by fine‑tuned model)
    prompt = (
        "Classify the dominant emotion in the following user message "
        "into one of [neutral, anxious, sad, angry, happy, stressed]. "
        "Just return the single word.\n\nUser: "
        f"{message}\nEmotion:"
    )
    mood = _llm.invoke(prompt).content.strip().lower()
    if mood not in {"neutral", "anxious", "sad", "angry", "happy", "stressed"}:
        mood = "neutral"
    return mood, "safe"


def retrieve_crisis_resource(locale: str) -> str:
    """Retrieve locale‑specific crisis hotline from the vector store."""
    try:
        docs = retriever.vectorstore.similarity_search(
            query=f"{locale} suicide prevention hotline", k=1, filter={"doc_type": "crisis_resource"}
        )
        return docs[0].page_content if docs else CRISIS_RESOURCE_FALLBACK
    except Exception as exc:  
        logger.warning("Failed to retrieve crisis resource: %s", exc)
        return CRISIS_RESOURCE_FALLBACK


def retrieve_skill_script(query: str) -> str:
    """Retrieve a coping skill script matching query (mood/goal)."""
    docs = retriever.vectorstore.similarity_search(
        query=query,
        k=1,
        filter={"doc_type": "skill_script"},
    )
    return docs[0].page_content if docs else "Let's practice slow, deep breathing together."  # fallback


def retrieve_reframe_template(distortion_label: str) -> str:
    """Retrieve Socratic questions / examples for the given cognitive distortion."""
    docs = retriever.vectorstore.similarity_search(
        query=f"Socratic questions for {distortion_label}",
        k=1,
        filter={"doc_type": "reframe_template"},
    )
    return docs[0].page_content if docs else "Could there be another way to interpret this situation?"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def detect_emotion_node(state: ChatState) -> ChatState:
    user_msg = state["last_user_msg"]
    mood, risk_level = classify_emotion_and_risk(user_msg)
    return {"mood": mood, "risk_level": risk_level}


def crisis_path_node(state: ChatState) -> ChatState:
    locale = state.get("user_locale", "US")
    resource_text = retrieve_crisis_resource(locale)
    assistant_response = (
        f"I'm really sorry you're feeling like this. {resource_text} "
        "If you can, consider reaching out to a trusted friend or mental-health professional right now."
    )
    state["chat_history"].append(AIMessage(content=assistant_response))
    return {"end_session": True}


def session_initializer_node(state: ChatState) -> ChatState:
    """Runs at the start of every session to pull latest summary."""
    user_id = state["user_id"]
    docs = retriever.vectorstore.similarity_search(
        query="latest session summary",
        k=1,
        filter={"user_id": user_id, "doc_type": "session_summary"},
    )
    summary = docs[0].page_content if docs else ""
    if summary:
        state["chat_history"].append(
            AIMessage(content=f"Welcome back. Last time we talked about: {summary}\nHow have you been?")
        )
    return {"prior_summary": summary}


def nlp_parse_node(state: ChatState) -> ChatState:  # placeholder
    """In a production system you would use spaCy / Transformers here.

    For this skeleton we simply pass through.
    """
    return {}


def distortion_detector_node(state: ChatState) -> ChatState:
    prompt = (
        "Detect if the user's message contains a cognitive distortion. "
        "If so, respond with the label (e.g., catastrophizing, black‑and‑white, mind‑reading, should‑statement). "
        "If none, return 'none'.\n\nUser message:" f" {state['last_user_msg']}\nLabel:"
    )
    label = _llm.invoke(prompt).content.strip().lower()
    label = "" if label == "none" else label
    return {"detected_distortion": label}


def reframe_prompt_node(state: ChatState) -> ChatState:
    template = retrieve_reframe_template(state["detected_distortion"])
    reply = _llm.invoke(
        f"User said: {state['last_user_msg']}\n"
        f"Cognitive distortion: {state['detected_distortion']}\n"
        f"Use the following Socratic template to coach the user to re‑frame: {template}"
    ).content
    state["chat_history"].append(AIMessage(content=reply))
    return {}


def skill_planner_node(state: ChatState) -> ChatState:
    query = f"{state['mood']} coping skill short"
    script = retrieve_skill_script(query)
    return {"skill_script": script}


def guide_exercise_node(state: ChatState) -> ChatState:
    script = state["skill_script"]
    reply = (
        "Let's try this together. " + script + "\nWhen you're ready, let me know how that felt."  # noqa: E501
    )
    state["chat_history"].append(AIMessage(content=reply))
    return {}


def collect_feedback_node(state: ChatState) -> ChatState:
    """Waits for user feedback. In this skeleton we treat the next user
    message as feedback and store sentiment (placeholder logic)."""
    user_msg = state["last_user_msg"]
    sentiment_prompt = (
        "Classify the sentiment of the following feedback as positive, neutral, or negative.\n" f"Feedback: {user_msg}\nSentiment:"
    )
    sentiment = _llm.invoke(sentiment_prompt).content.strip().lower()
    return {"feedback_sentiment": sentiment}


def adjust_instruction_node(state: ChatState) -> ChatState:
    sentiment = state.get("feedback_sentiment", "neutral")
    if sentiment == "negative":
        reply = (
            "Thanks for letting me know. Let's try an alternative approach—perhaps a grounding exercise using your senses. "
            "Focus on naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste."
        )
    else:
        reply = "Great job! We'll keep practicing this skill as it seems helpful."
    state["chat_history"].append(AIMessage(content=reply))
    return {}


def summary_writer_node(state: ChatState) -> ChatState:
    summary = _llm.invoke(
        "Summarise the key points and progress of the following conversation in 3 sentences:\n" + "\n".join(m.content for m in state["chat_history"])  # type: ignore[arg-type]
    ).content

    # Persist into vector DB
    retriever.vectorstore.add_documents(
        [
            {
                "page_content": summary,
                "metadata": {
                    "user_id": state["user_id"],
                    "timestamp": str(int(time.time())),
                    "doc_type": "session_summary",
                },
            }
        ]
    )
    logger.info("Session summary stored. Length: %d chars", len(summary))
    return {}

# ---------------------------------------------------------------------------
# Edge guards
# ---------------------------------------------------------------------------

def is_crisis(state: ChatState) -> bool:
    return state.get("risk_level") == "crisis"


def has_distortion(state: ChatState) -> bool:
    return bool(state.get("detected_distortion"))


def needs_adjust(state: ChatState) -> bool:
    return state.get("feedback_sentiment") == "negative"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph[ChatState]:
    sg: StateGraph[ChatState] = StateGraph(ChatState)

    # 1. Add nodes
    sg.add_node("detect_emotion", detect_emotion_node)
    sg.add_node("crisis_path", crisis_path_node)
    sg.add_node("session_initializer", session_initializer_node)
    sg.add_node("nlp_parse", nlp_parse_node)
    sg.add_node("distortion_detector", distortion_detector_node)
    sg.add_node("reframe_prompt", reframe_prompt_node)
    sg.add_node("skill_planner", skill_planner_node)
    sg.add_node("guide_exercise", guide_exercise_node)
    sg.add_node("collect_feedback", collect_feedback_node)
    sg.add_node("adjust_instruction", adjust_instruction_node)
    sg.add_node("summary_writer", summary_writer_node)

    # 2. Wiring edges
    sg.add_edge(START, "detect_emotion")

    # Crisis routing
    sg.add_conditional_edges(
        "detect_emotion",
        {
            "crisis_path": is_crisis,
            "session_initializer": lambda _s: True,
        },
    )

    sg.add_edge("crisis_path", END)

    # Main flow
    sg.add_edge("session_initializer", "nlp_parse")
    sg.add_edge("nlp_parse", "distortion_detector")

    sg.add_conditional_edges(
        "distortion_detector",
        {
            "reframe_prompt": has_distortion,
            "skill_planner": lambda _s: True,
        },
    )

    sg.add_edge("reframe_prompt", "skill_planner")
    sg.add_edge("skill_planner", "guide_exercise")
    sg.add_edge("guide_exercise", "collect_feedback")

    sg.add_conditional_edges(
        "collect_feedback",
        {
            "adjust_instruction": needs_adjust,
            "summary_writer": lambda _s: True,
        },
    )

    sg.add_edge("adjust_instruction", "guide_exercise")
    sg.add_edge("summary_writer", END)

    graph = sg.compile()

    with open("graph_output.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    return graph