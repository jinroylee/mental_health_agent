""" Mental health assistant graph.
"""

import json
import logging
import os
import time
from pathlib import Path

import dotenv
from openai import OpenAI
from langchain.chains.moderation import OpenAIModerationChain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from psycopg import Connection

from graphs.prompts import *
from graphs.types import ChatState, Diagnosis
from graphs.utils import _similarity_search, retrieve_crisis_resource, retrieve_reframe_template, retrieve_counseling_resource, retrieve_therapy_script
from graphs.pretty_logging import get_pretty_logger
from graphs.langchains import *
from graphs.shared_config import embed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pretty_logger = get_pretty_logger(__name__)

# ---------------------------------------------------------------------------
# Environment & shared resources
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["INDEX_NAME"]
POSTGRES_DB_URI = os.getenv("POSTGRES_DB_URI")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

conn = Connection.connect(POSTGRES_DB_URI, **connection_kwargs)

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
similarity_retriever = PineconeVectorStore(index=index, embedding=embed).as_retriever(
    search_kwargs={
        "k": 1
    }
)
base_retriever = PineconeVectorStore(index=index, embedding=embed).as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 40,
        "lambda_mult": 0.25,
    }
)

compressor = EmbeddingsFilter(embeddings=embed, similarity_threshold=0.30)
mmr_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever, base_compressor=compressor
)

moderation_chain = OpenAIModerationChain()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# Keep only the last K messages in working memory to control token usage
TRIM_HISTORY_MAX_MESSAGES = int(os.getenv("TRIM_HISTORY_MAX_MESSAGES", "5"))

def _trim_chat_history_in_place(state: ChatState) -> None:
    if not state.get("chat_history"):
        return
    if TRIM_HISTORY_MAX_MESSAGES > 0 and len(state["chat_history"]) > TRIM_HISTORY_MAX_MESSAGES:
        print(f"trimming chat history from {len(state['chat_history'])} to {TRIM_HISTORY_MAX_MESSAGES}")
        state["chat_history"] = state["chat_history"][-TRIM_HISTORY_MAX_MESSAGES:]

# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def detect_emotion_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("detect_emotion_node")
    user_msg = state["last_user_msg"]# Add user message to chat history
    
    state["chat_history"].append(HumanMessage(content=user_msg))
    mod_res = openai_client.moderations.create(model="text-moderation-latest", input=user_msg).model_dump()["results"][0]
    state["risk_level"] = "crisis" if mod_res["flagged"] and mod_res["categories"].get("self_harm", False) else "safe"
    
    mood_raw = mood_chain.invoke({"text": user_msg}).strip().lower()
    state["mood"] = mood_raw if mood_raw in {"happy", "sad", "anxious", "angry", "neutral", "stressed"} else "neutral"  # type: ignore
    pretty_logger.state_print("mood", state["mood"])
    pretty_logger.state_print("risk_level", state["risk_level"])
    pretty_logger.node_separator_bottom("detect_emotion_node")
    return state


def crisis_path_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("crisis_path_node")
    locale = state.get("user_locale", "US")
    user_msg = state["last_user_msg"]

    # Retrieve crisis resources from vector DB with improved formatting
    resource_text = retrieve_crisis_resource(mmr_retriever, locale)

    pretty_logger.state_print("resource_text", resource_text)
    pretty_logger.state_print("user_msg", user_msg)

    # Generate response using crisis chain
    assistant_response = crisis_chain.invoke({
        "resources": resource_text,
        "user_message": user_msg
    })

    state.setdefault("chat_history", []).append(AIMessage(content=assistant_response))
    pretty_logger.node_separator_bottom("crisis_path_node")
    return state


def session_initializer_node(state: ChatState) -> ChatState:
    """Runs at the start of every session to pull latest summary."""
    pretty_logger.node_separator_top("session_initializer_node")
    pretty_logger.state_print("user_id", state["user_id"])
    user_msg = state["last_user_msg"]
    user_id = state["user_id"]
    # Ensure chat_history is initialized before any appends
    state.setdefault("chat_history", [])
    # Get all session summaries for this user, then find the most recent by timestamp
    docs = _similarity_search(
        similarity_retriever,
        query="session summary",
        filters={"user_id": user_id, "doc_type": "session_summary", "thread_id": state.get("thread_id", "")},
        k=10,  # Get more documents to ensure we capture all sessions
    )
    pretty_logger.state_print("docs", docs)

    # Sort by timestamp (newest first) and get the most recent
    if docs:
        sorted_docs = sorted(docs, key=lambda d: int(d.metadata.get("timestamp", "0")), reverse=True)
        summary = sorted_docs[0].page_content
        pretty_logger.state_print("selected_latest_summary", summary)
    else:
        summary = ""
    # if summary:
    #     state["chat_history"].append(
    #         AIMessage(content=f"Welcome back. Last time we talked about: {summary}\nHow have you been?")
    #     )
        
    _trim_chat_history_in_place(state)
    
    pretty_logger.state_print("Chat history", state["chat_history"])
    pretty_logger.node_separator_bottom("session_initializer_node")
    state.update(prior_summary=summary)
    return state

#classify if this is a feedback or not. If feedback, collect feedback. If not, continue with the conversation.
def classify_feedback_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("classify_feedback_node")
    feedback_type = classify_feedback_chain.invoke({"input": state["last_user_msg"]})
    pretty_logger.state_print("feedback_type", feedback_type)
    state.update(is_feedback=feedback_type == "feedback")
    pretty_logger.node_separator_bottom("classify_feedback_node")
    return state

def diagnose_node(state: ChatState) -> ChatState:
    """Diagnose the user's mental health condition."""
    pretty_logger.node_separator_top("diagnose_node")
    user_msg = state["last_user_msg"]
    
    try:
        data = diagnosis_chain.invoke({
            "history": state.get("chat_history", []), 
            "input": user_msg
        })
        pretty_logger.state_print("data", data)
        needs_therapy = bool(data.get("needs_therapy", False))
        diagnosis: Diagnosis = data.get("diagnosis", "none")  # type: ignore[assignment]
    except (json.JSONDecodeError, Exception) as e:
        pretty_logger.warning("Diagnose JSON parse error: %s", e)
        needs_therapy, diagnosis = False, "none"

    state.update(needs_therapy=needs_therapy, diagnosis=diagnosis)  # type: ignore[arg-type]
    pretty_logger.node_separator_bottom("diagnose_node")
    return state


def counseling_dialogue_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("counseling_dialogue_node")
    pretty_logger.state_print("state", state)

    query = state["last_user_msg"]
    
    # Retrieve counseling resources with improved context formatting
    resource = retrieve_counseling_resource(mmr_retriever, query)
    
    # Check if we have meaningful context
    has_relevant_context = "No specific counseling resources found" not in resource
    pretty_logger.state_print("has_relevant_context", has_relevant_context)
    
    answer = counseling_chain.invoke({
        "ctx": resource,
        "prior_summary": state.get("prior_summary", ""),
        "question": query,
        "history": state.get("chat_history", []),
    })
    pretty_logger.state_print("answer", answer)
    state.setdefault("chat_history", []).append(AIMessage(content=answer))
    pretty_logger.state_print("chat_history", state["chat_history"])
    pretty_logger.node_separator_bottom("counseling_dialogue_node")
    return state

def nlp_parse_node(state: ChatState) -> ChatState: 
    """In a production system you would use spaCy / Transformers here.

    For this skeleton we simply pass through.
    """
    pretty_logger.node_separator_top("nlp_parse_node")
    pretty_logger.state_print("state", state)
    pretty_logger.node_separator_bottom("nlp_parse_node")
    return {}

def distortion_detector_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("distortion_detector_node")
    pretty_logger.state_print("last_user_msg", state["last_user_msg"])
    
    label = distortion_chain.invoke({"message": state['last_user_msg']}).strip().lower()
    pretty_logger.state_print("label", label)
    label = None if label == "none" else label
    pretty_logger.node_separator_bottom("distortion_detector_node")
    return {"detected_distortion": label}

def reframe_prompt_node(state: ChatState) -> ChatState:
    """
    Reframe the user's message based on the detected distortion.
    """
    pretty_logger.node_separator_top("reframe_prompt_node")
    pretty_logger.state_print("last_user_msg", state["last_user_msg"])
    pretty_logger.state_print("detected_distortion", state["detected_distortion"])
    distortion = state["detected_distortion"]
    
    # Retrieve reframing template with improved error handling
    template = retrieve_reframe_template(mmr_retriever, distortion) if distortion else ""
    
    # Check if we have a specific template or are using fallback
    has_specific_template = distortion and "No specific reframing template found" not in template
    pretty_logger.state_print("has_specific_template", has_specific_template)
    pretty_logger.state_print("template", template)
    
    reply = reframe_chain.invoke({
        "tmpl": template,
        "prior_summary": state.get("prior_summary", ""),
        "u": state["last_user_msg"],
    })
    pretty_logger.state_print("reply", reply)
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    pretty_logger.node_separator_bottom("reframe_prompt_node")
    return state

def therapy_planner_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("therapy_planner_node")
    diagnosis: Diagnosis = state.get("diagnosis", "none")
    
    # Use diagnosis for more targeted therapy script retrieval
    query = f"{diagnosis} coping strategies" if diagnosis != "none" else "general mental health coping"
    therapy_script = retrieve_therapy_script(mmr_retriever, query)
    
    # Check if we have specific therapy scripts or are using fallback
    has_specific_script = "No specific therapy scripts found" not in therapy_script
    pretty_logger.state_print("has_specific_script", has_specific_script)
    pretty_logger.state_print("therapy_script", therapy_script)
    
    state.update(therapy_script=therapy_script, therapy_attempts=0)
    pretty_logger.node_separator_bottom("therapy_planner_node")
    return state


def guide_exercise_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("guide_exercise_node")
    script = state.get("therapy_script", "Let's begin a breathing exercise…")
    reply = guide_exercise_chain.invoke({
        "script": script,
        "prior_summary": state.get("prior_summary", ""),
        "u": state["last_user_msg"],
    })
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    pretty_logger.state_print("Chat history", state["chat_history"])
    state["therapy_attempts"] = state.get("therapy_attempts", 0) + 1  # type: ignore[index]
    pretty_logger.node_separator_bottom("guide_exercise_node")
    return state


def collect_feedback_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("collect_feedback_node")
    user_msg = state["last_user_msg"]
    pretty_logger.state_print("user_msg", user_msg)
    sentiment_raw = sentiment_chain.invoke({"feedback": user_msg}).strip().lower()
    sentiment = sentiment_raw if sentiment_raw in {"positive", "neutral", "negative"} else "neutral"
    state.update(feedback_sentiment=sentiment)  # type: ignore[arg-type]
    pretty_logger.node_separator_bottom("collect_feedback_node")
    return state


def adjust_instruction_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("adjust_instruction_node")

    reply = adjust_instruction_chain.invoke({"u": state["last_user_msg"]})
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    pretty_logger.state_print("Chat history", state["chat_history"])
    pretty_logger.node_separator_bottom("adjust_instruction_node")
    return state


def summary_writer_node(state: ChatState) -> ChatState:
    pretty_logger.node_separator_top("summary_writer_node")
    history = state.get("chat_history", [])
    user_id = state["user_id"]

    # Window size for working memory; only summarize when we exceed this size
    try:
        window_size = TRIM_HISTORY_MAX_MESSAGES
    except Exception:
        window_size = 5

    # If we are within the window, skip summarization to save tokens
    if window_size <= 0 or len(history) <= window_size:
        pretty_logger.info("History within window (len=%d, window=%d); skipping summary update.", len(history), window_size)
        pretty_logger.node_separator_bottom("summary_writer_node")
        return {}

    # Fold only the oldest message into the rolling summary each turn
    oldest_messages = history[:len(history) - window_size]
    oldest_text = "\n".join([f"User: {message.content}" if isinstance(message, HumanMessage) else f"Assistant: {message.content}" for message in oldest_messages])

    # Build input for updating rolling summary using existing chain
    existing_summary = state.get("prior_summary", "")
    conv_for_update = (
        (f"Existing summary:\n{existing_summary}\n\n" if existing_summary else "")
        + "New message to incorporate into the summary:\n"
        + oldest_text
    )
    pretty_logger.state_print("rolling_summary_input", conv_for_update)
    new_summary = summary_chain.invoke({"conv": conv_for_update})
    pretty_logger.state_print("new_summary", new_summary)

    # Remove the oldest message so the working window slides by one per turn
    state["chat_history"] = history[1:]
    state["prior_summary"] = new_summary

    # Persist the updated rolling summary (keep one per user)
    try:
        existing_docs = _similarity_search(
            mmr_retriever,
            query="session summary",
            filters={"user_id": user_id, "doc_type": "session_summary"},
            k=10,
        )
        if existing_docs:
            vector_ids = []
            for doc in existing_docs:
                if hasattr(doc, 'id') and doc.id:
                    vector_ids.append(doc.id)
            if vector_ids:
                index.delete(ids=vector_ids)
                pretty_logger.info("Deleted %d existing session summaries for user %s", len(vector_ids), user_id)
            else:
                try:
                    index.delete(filter={"user_id": user_id, "doc_type": "session_summary"})
                    pretty_logger.info("Deleted existing session summaries using metadata filter for user %s", user_id)
                except Exception as e:
                    pretty_logger.warning("Could not delete by filter: %s", e)
    except Exception as e:
        pretty_logger.warning("Error deleting existing summaries: %s", e)

    doc = Document(
        page_content=new_summary,
        metadata={
            "user_id": user_id,
            "thread_id": state.get("thread_id", ""),
            "timestamp": str(int(time.time())),
            "doc_type": "session_summary",
        }
    )
    mmr_retriever.vectorstore.add_documents([doc])
    pretty_logger.info("Updated session summary stored. Length: %d chars", len(new_summary))
    pretty_logger.node_separator_bottom("summary_writer_node")
    # Return full state so chat_history/prior_summary changes are captured
    return state

# ---------------------------------------------------------------------------
# Edge guards
# ---------------------------------------------------------------------------

def is_crisis(state: ChatState) -> bool:
    return state.get("risk_level") == "crisis"

def has_distortion(state: ChatState) -> bool:
    return bool(state.get("detected_distortion"))

def needs_adjust(state: ChatState) -> bool:
    return state.get("feedback_sentiment") == "negative"

def needs_therapy_script(state: ChatState) -> bool:
    return state.get("needs_therapy")

def is_feedback(state: ChatState) -> bool:
    return state.get("is_feedback")

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph[ChatState]:
    sg: StateGraph[ChatState] = StateGraph(ChatState)

    # 1. Add nodes
    sg.add_node("detect_emotion", detect_emotion_node)
    sg.add_node("crisis_path", crisis_path_node)
    sg.add_node("session_initializer", session_initializer_node)
    sg.add_node("classify_feedback", classify_feedback_node)
    sg.add_node("diagnose", diagnose_node)
    sg.add_node("counseling_dialogue", counseling_dialogue_node)
    sg.add_node("nlp_parse", nlp_parse_node)
    sg.add_node("distortion_detector", distortion_detector_node)
    sg.add_node("reframe_prompt", reframe_prompt_node)
    sg.add_node("therapy_planner", therapy_planner_node)
    sg.add_node("guide_exercise", guide_exercise_node)
    sg.add_node("collect_feedback", collect_feedback_node)
    sg.add_node("adjust_instruction", adjust_instruction_node)
    sg.add_node("summary_writer", summary_writer_node)

    # 2. Wiring edges
    sg.add_edge(START, "detect_emotion")

    # Crisis routing
    sg.add_conditional_edges(
        "detect_emotion",
        is_crisis,
        {
            True: "crisis_path",
            False: "session_initializer",
        },
    )

    sg.add_edge("crisis_path", "summary_writer")

    # Main flow
    sg.add_edge("session_initializer", "classify_feedback")

    sg.add_conditional_edges(
        "classify_feedback",
        is_feedback,
        {
            True: "collect_feedback",
            False: "diagnose",
        },
    )

    sg.add_conditional_edges(
        "diagnose",
        needs_therapy_script,
        {
            True: "nlp_parse",
            False: "counseling_dialogue",
        },
    )

    sg.add_edge("counseling_dialogue", "summary_writer")

    sg.add_edge("nlp_parse", "distortion_detector")

    sg.add_conditional_edges(
        "distortion_detector",
         has_distortion,
        {
            False: "therapy_planner",
            True: "reframe_prompt",
        },
    )

    sg.add_edge("reframe_prompt", "therapy_planner")
    sg.add_edge("therapy_planner", "guide_exercise")

    sg.add_conditional_edges(
        "collect_feedback",
        needs_adjust,
        {
            True : "adjust_instruction",
            False : "summary_writer",
        },
    )

    sg.add_edge("adjust_instruction", "summary_writer")

    sg.add_edge("guide_exercise", "summary_writer")        # stop after giving the exercise
    sg.add_edge("summary_writer", END)

    # Compile with Postgres checkpointer if configured
    if not POSTGRES_DB_URI:
        logger.warning("POSTGRES_DB_URI not set; graph will run without persistent checkpoints.")
        graph = sg.compile()
    else:
        # Dynamically import PostgresSaver to avoid hard dependency at import-time
        try:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            graph = sg.compile(checkpointer=checkpointer)
        except Exception as exc:
            logger.warning("Failed to initialize Postgres checkpointer (%s); running without persistence.", exc)
            graph = sg.compile()

    with open("graph_output.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    return graph