from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

import dotenv
from langchain.chains.moderation import OpenAIModerationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.3)
_embed = OpenAIEmbeddings(model="text-embedding-ada-002")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
retriever = PineconeVectorStore(index=index, embedding=_embed).as_retriever(
    search_kwargs={"k": 4}
)

moderation_chain = OpenAIModerationChain()
conversation_memory = ConversationBufferMemory(return_messages=True)

CRISIS_RESOURCE_FALLBACK = (
    "If you believe you may harm yourself or others, please reach out for immediate help. "
    "In the U.S. call 988 or visit https://988lifeline.org. If you are outside the U.S., "
    "search online for a local crisis helpline in your country. You are not alone and help is available."
)

SYSTEM_PROMPT = (
    "You are a supportive mental‑health assistant.\n"
    "Use the provided context to answer with empathy and practical advice.\n"
    "If the user requests medical diagnosis or expresses intent to self‑harm, "
    "provide crisis resources and encourage professional help."
)

Mood = Literal["happy", "sad", "anxious", "angry", "neutral", "stressed"]
Diagnosis = Literal["depression", "anxiety", "none"]
RiskLevel = Literal["safe", "crisis"]

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class ChatState(TypedDict, total=False):
    """Mutable state that flows through the LangGraph."""

    # Static / session‑level
    user_id: str
    user_locale: str

    # Rolling conversation buffer
    chat_history: List[BaseMessage]

    # Safety / affect
    mood: Mood
    risk_level: Literal["safe", "crisis"]

    # Diagnosis + routing
    diagnosis: Diagnosis
    needs_skill: bool  # branch flag for skill vs. knowledge

    # Cognitive distortion
    detected_distortion: Optional[str]

    # Skill loop
    skill_script: Optional[str]
    skill_attempts: int
    feedback_sentiment: Optional[str]

    # Summaries
    prior_summary: Optional[str]

    # Last user message (set each turn)
    last_user_msg: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _similarity_search(query: str, *, filters: Optional[dict] = None, k: int = 3):
    return retriever.vectorstore.similarity_search(query=query, k=k, filter=filters or {})

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

def retrieve_reframe_template(distortion_label: str) -> str:
    docs = _similarity_search(
        query=f"Socratic questions for {distortion_label}",
        filters={"doc_type": "reframe_template", "distortion_label": distortion_label},
        k=1,
    )
    return docs[0].page_content if docs else "Could there be another perspective on this?"

def retrieve_skill_script(query: str) -> str:
    """Retrieve a coping skill script matching query (mood/goal)."""
    docs = retriever.vectorstore.similarity_search(
        query=query,
        k=1,
        filter={"doc_type": "skill_script"},
    )
    return docs[0].page_content if docs else "Let's practice slow, deep breathing together."  # fallback


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def detect_emotion_node(state: ChatState) -> ChatState:
    user_msg = state["last_user_msg"]
    mood, risk_level = classify_emotion_and_risk(user_msg)
    print("=============================================detect_emotion_node ============================================================")
    print("user_msg: ", user_msg)
    print("mood: ", mood)
    print("risk_level: ", risk_level)
    print("=============================================detect_emotion_node ============================================================")
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
    print("=============================================session_initializer_node ============================================================")
    print("user_id: ", state["user_id"])
    user_id = state["user_id"]
    docs = retriever.vectorstore.similarity_search(
        query="latest session summary",
        k=1,
        filter={"user_id": user_id, "doc_type": "session_summary"},
    )
    print("docs: ", docs)
    summary = docs[0].page_content if docs else ""
    print("summary: ", summary)
    if summary:
        state["chat_history"].append(
            AIMessage(content=f"Welcome back. Last time we talked about: {summary}\nHow have you been?")
        )
    print("prior_summary: ", summary)
    print("=============================================session_initializer_node ============================================================")
    return {"prior_summary": summary}

def diagnose_node(state: ChatState) -> ChatState:
    user_msg = state["last_user_msg"]
    diag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a triage assistant. Analyse the user's latest message **and** prior summary if available.
            1. Decide if the user is asking for coping skills (`needs_skill=true`) or merely wants to discuss / learn.
            2. If skills are needed, guess the main condition: depression, anxiety, none.
            Return ONLY valid JSON: {\"needs_skill\": bool, \"diagnosis\": \"depression|anxiety|none\"}
            """,
        ),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
    ])
    raw = llm.invoke(diag_prompt.format(history=state.get("chat_history", []), input=user_msg)).content
    try:
        data = json.loads(raw)
        needs_skill = bool(data.get("needs_skill", False))
        diagnosis: Diagnosis = data.get("diagnosis", "none")  # type: ignore[assignment]
    except json.JSONDecodeError:
        logger.warning("Diagnose JSON parse error: %s", raw)
        needs_skill, diagnosis = False, "none"

    state.update(needs_skill=needs_skill, diagnosis=diagnosis)  # type: ignore[arg-type]
    return state


def knowledge_dialogue_node(state: ChatState) -> ChatState:
    query = state["last_user_msg"]
    docs = _similarity_search(query, filters={"doc_type": "psycho_education", "locale": "en"}, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\nYou are having an exploratory conversation; teach in clear, non‑clinical language."),
        ("system", "Context:\n{ctx}"),
        ("user", "{question}"),
    ])
    answer = llm.invoke(prompt.format(ctx=context, question=query)).content
    state.setdefault("chat_history", []).append(AIMessage(content=answer))
    return state

def nlp_parse_node(state: ChatState) -> ChatState:  # placeholder
    """In a production system you would use spaCy / Transformers here.

    For this skeleton we simply pass through.
    """
    print("=============================================nlp_parse_node ============================================================")
    print("state: ", state)
    print("=============================================nlp_parse_node ============================================================")
    return {}


def distortion_detector_node(state: ChatState) -> ChatState:
    print("=============================================distortion_detector_node ============================================================")
    print("last_user_msg: ", state["last_user_msg"])
    messages = [
    (
        "system",
        "Detect if the user's message contains a cognitive distortion. "
        "If so, respond with the label (e.g., catastrophizing, black-and-white, mind-reading, should-statement). "
        "If none, return 'none'.User message:",
    ),
    ("human", state['last_user_msg']),
    ]

    print("messages: ", messages)
    print("llm.invoke(messages): ", llm.invoke(messages))
    print("llm response: ", llm.invoke(messages).content)

    label = llm.invoke(messages).content.strip().lower()
    print("label: ", label)
    label = None if label == "none" else label
    print("=============================================distortion_detector_node ============================================================")
    return {"detected_distortion": label}

def reframe_prompt_node(state: ChatState) -> ChatState:
    distortion = state["detected_distortion"]
    template = retrieve_reframe_template(distortion) if distortion else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("system", "Template:\n{tmpl}"),
        ("user", "User said: {u}\nPlease respond with Socratic coaching."),
    ])
    reply = llm.invoke(prompt.format(tmpl=template, u=state["last_user_msg"])).content
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    return state


def skill_planner_node(state: ChatState) -> ChatState:
    print("=============================================skill_planner_node ============================================================")
    query = f"{state['mood']} coping skill short"
    script = retrieve_skill_script(query)
    return {"skill_script": script}


def skill_planner_node(state: ChatState) -> ChatState:
    diagnosis: Diagnosis = state.get("diagnosis", "none")
    filters = {"doc_type": "skill_script"}
    if diagnosis != "none":
        filters["skill_diagnosis"] = diagnosis
    docs = _similarity_search(
        query=f"coping skill script for {diagnosis}" if diagnosis != "none" else "general coping skill script",
        filters=filters,
        k=3,
    )
    script = "\n\n".join(d.page_content for d in docs) or "Let's do a simple grounding exercise: notice 5 things you can see…"
    state.update(skill_script=script, skill_attempts=0)  # type: ignore[arg-type]
    return state


def guide_exercise_node(state: ChatState) -> ChatState:
    script = state.get("skill_script", "Let's begin a breathing exercise…")
    intro = script.split("\n")[0]
    reply = f"Let's try this together. {intro}\nWhen you're ready, let me know how that felt."
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    state["skill_attempts"] = state.get("skill_attempts", 0) + 1  # type: ignore[index]
    return state

# 11) Collect feedback --------------------------------------------------------

def collect_feedback_node(state: ChatState) -> ChatState:
    user_msg = state["last_user_msg"]
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the sentiment of the feedback as positive, neutral, or negative."),
        ("user", "{f}"),
    ])
    sentiment_raw = llm.invoke(sentiment_prompt.format(f=user_msg)).content.strip().lower()
    sentiment = sentiment_raw if sentiment_raw in {"positive", "neutral", "negative"} else "neutral"
    state.update(feedback_sentiment=sentiment)  # type: ignore[arg-type]
    return state


def adjust_instruction_node(state: ChatState) -> ChatState:
    sentiment = state.get("feedback_sentiment", "neutral")
    if sentiment == "negative":
        reply = (
            "Thanks for letting me know. Let's try an alternative approach—perhaps a sensory grounding exercise. "
            "Focus on naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste."
        )
    else:
        reply = "Great job! We'll keep practicing this skill since it seems helpful."
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    return state


def summary_writer_node(state: ChatState) -> ChatState:
    convo = "\n".join(m.content for m in state.get("chat_history", []))

    summary = llm.invoke(
        ChatPromptTemplate.from_messages([
            ("system", "Summarise the key points and progress of the following conversation in 3 sentences."),
            ("system", "{conv}"),
        ]).format(conv=convo)
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

def needs_skill_script(state: ChatState) -> bool:
    return state.get("needs_skill")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph[ChatState]:
    sg: StateGraph[ChatState] = StateGraph(ChatState)

    # 1. Add nodes
    sg.add_node("detect_emotion", detect_emotion_node)
    sg.add_node("crisis_path", crisis_path_node)
    sg.add_node("session_initializer", session_initializer_node)
    sg.add_node("diagnose", diagnose_node)
    sg.add_node("knowledge_dialogue", knowledge_dialogue_node)
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
        is_crisis,
        {
            True: "crisis_path",
            False: "session_initializer",
        },
    )

    sg.add_edge("crisis_path", END)

    # Main flow
    sg.add_edge("session_initializer", "diagnose")

    sg.add_conditional_edges(
        "diagnose",
        needs_skill_script,
        {
            True: "nlp_parse",
            False: "knowledge_dialogue",
        },
    )

    sg.add_edge("nlp_parse", "distortion_detector")

    sg.add_conditional_edges(
        "distortion_detector",
         has_distortion,
        {
            False: "skill_planner",
            True: "reframe_prompt",
        },
    )

    sg.add_edge("reframe_prompt", "skill_planner")
    sg.add_edge("skill_planner", "guide_exercise")
    sg.add_edge("guide_exercise", "collect_feedback")

    sg.add_conditional_edges(
        "collect_feedback",
        needs_adjust,
        {
            True : "adjust_instruction",
            False : "summary_writer",
        },
    )

    sg.add_edge("adjust_instruction", "guide_exercise")
    sg.add_edge("summary_writer", END)

    graph = sg.compile()

    with open("graph_output.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    return graph