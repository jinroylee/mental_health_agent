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
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptValue
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from graphs.prompts import *
from graphs.types import ChatState, Diagnosis
from graphs.utils import _similarity_search, retrieve_crisis_resource, retrieve_reframe_template, retrieve_therapy_script

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

# ---------------------------------------------------------------------------
# Chain definitions
# ---------------------------------------------------------------------------
def log_llm_input(messages: ChatPromptValue):
    """Log the formatted prompt that gets sent to the LLM."""
    print("=== FORMATTED PROMPT TO LLM ===")
    for msg in messages.to_messages():
        print(f"Role: {msg.type}")
        print(f"Content: {msg.content}")
        print("---")
    print("=== END FORMATTED PROMPT ===")
    return messages

def log_llm_output(output):
    """Log the raw LLM output before parsing."""
    logger.info("Raw LLM output for diagnosis: %s", output.content)
    print("=== RAW LLM OUTPUT ===")
    print(output.content)
    print("=== END RAW OUTPUT ===")
    return output

# Mood detection chain
mood_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return one word: happy, sad, anxious, angry, neutral, or stressed."),
    ("user", "{text}"),
])
mood_chain = mood_prompt | llm | StrOutputParser()

# Diagnosis chain with JSON output parser
diagnosis_prompt = ChatPromptTemplate.from_messages([
   ( "system", DIAGNOSIS_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
])

diagnosis_parser = JsonOutputParser()
diagnosis_chain = diagnosis_prompt | llm | diagnosis_parser

# Distortion detection chain
distortion_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Detect if the user's message contains a cognitive distortion. "
        "If so, respond with the label (e.g., catastrophizing, black-and-white, mind-reading, should-statement). "
        "If none, return 'none'."
    ),
    ("human", "{message}"),
])
distortion_chain = distortion_prompt | llm | StrOutputParser()

# Sentiment analysis chain
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the sentiment of the feedback as positive, neutral, or negative."),
    ("user", "{feedback}"),
])
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# Crisis response chain
crisis_prompt = ChatPromptTemplate.from_messages([
    ("system", CRISIS_SYSTEM_PROMPT),
    ("system", "Crisis Resources Available:\n{resources}"),
    ("user", "{user_message}"),
])
crisis_chain = crisis_prompt | llm | StrOutputParser()

# Counseling dialogue chain
counseling_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nYou are having an exploratory conversation; teach in clear, non‑clinical language."),
    ("system", "Context:\n{ctx}"),
    ("user", "{question}"),
])
counseling_chain = counseling_prompt | llm | StrOutputParser()

# Reframe response chain
reframe_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "Template:\n{tmpl}"),
    ("user", "User said: {u}\nPlease respond with Socratic coaching."),
])
reframe_chain = reframe_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Summary generation chain
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the key points and progress of the following conversation in 3 sentences."),
    ("system", "{conv}"),
])
summary_chain = summary_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def detect_emotion_node(state: ChatState) -> ChatState:
    logger.info("#########detect_emotion_node#########")
    user_msg = state["last_user_msg"]
    mod_res = openai_client.moderations.create(model="text-moderation-latest", input=user_msg).model_dump()["results"][0]
    state["risk_level"] = "crisis" if mod_res["flagged"] and mod_res["categories"].get("self_harm", False) else "safe"
    
    mood_raw = mood_chain.invoke({"text": user_msg}).strip().lower()
    state["mood"] = mood_raw if mood_raw in {"happy", "sad", "anxious", "angry", "neutral", "stressed"} else "neutral"  # type: ignore
    print("mood: ", state["mood"])
    print("risk_level: ", state["risk_level"])
    logger.info("#########detect_emotion_node#########")
    return state


def crisis_path_node(state: ChatState) -> ChatState:
    logger.info("#########crisis_path_node#########")
    locale = state.get("user_locale", "US")
    user_msg = state["last_user_msg"]

    # Retrieve crisis resources from vector DB
    resource_text = retrieve_crisis_resource(retriever, locale)

    print("resource_text: ", resource_text)
    print("user_msg: ", user_msg)

    # Generate response using crisis chain
    assistant_response = crisis_chain.invoke({
        "resources": resource_text,
        "user_message": user_msg
    })

    state["chat_history"].append(AIMessage(content=assistant_response))
    logger.info("#########crisis_path_node#########")
    return {"end_session": True}


def session_initializer_node(state: ChatState) -> ChatState:
    """Runs at the start of every session to pull latest summary."""
    print("#########session_initializer_node#########")
    print("user_id: ", state["user_id"])
    user_id = state["user_id"]
    docs = _similarity_search(
        retriever,
        query="latest session summary",
        filters={"user_id": user_id, "doc_type": "session_summary"},
        k=1,
    )
    print("docs: ", docs)
    summary = docs[0].page_content if docs else ""
    print("summary: ", summary)
    if summary:
        state["chat_history"].append(
            AIMessage(content=f"Welcome back. Last time we talked about: {summary}\nHow have you been?")
        )
    print("prior_summary: ", summary)
    print("#########session_initializer_node#########")
    state.update(prior_summary=summary)
    return state

def diagnose_node(state: ChatState) -> ChatState:
    """Diagnose the user's mental health condition."""
    print("#########diagnose_node#########")
    user_msg = state["last_user_msg"]
    print("user_msg: ", user_msg)
    
    try:
        data = diagnosis_chain.invoke({
            "history": state.get("chat_history", []), 
            "input": user_msg
        })
        print("data: ", data)
        needs_therapy = bool(data.get("needs_therapy", False))
        diagnosis: Diagnosis = data.get("diagnosis", "none")  # type: ignore[assignment]
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Diagnose JSON parse error: %s", e)
        needs_therapy, diagnosis = False, "none"

    state.update(needs_therapy=needs_therapy, diagnosis=diagnosis)  # type: ignore[arg-type]
    print("#########diagnose_node#########")
    return state


def counseling_dialogue_node(state: ChatState) -> ChatState:
    query = state["last_user_msg"]
    docs = _similarity_search(retriever, query, filters={"doc_type": "counseling_resource", "locale": "en"}, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    
    answer = counseling_chain.invoke({
        "ctx": context, 
        "question": query
    })
    
    state.setdefault("chat_history", []).append(AIMessage(content=answer))
    return state

def nlp_parse_node(state: ChatState) -> ChatState: 
    """In a production system you would use spaCy / Transformers here.

    For this skeleton we simply pass through.
    """
    print("#########nlp_parse_node#########")
    print("state: ", state)
    print("#########nlp_parse_node#########")
    return {}

def distortion_detector_node(state: ChatState) -> ChatState:
    print("#########distortion_detector_node#########")
    print("last_user_msg: ", state["last_user_msg"])
    
    label = distortion_chain.invoke({"message": state['last_user_msg']}).strip().lower()
    print("label: ", label)
    label = None if label == "none" else label
    print("#########distortion_detector_node#########")
    return {"detected_distortion": label}

def reframe_prompt_node(state: ChatState) -> ChatState:
    """
    Reframe the user's message based on the detected distortion.
    """
    print("#########reframe_prompt_node#########")
    print("last_user_msg: ", state["last_user_msg"])
    print("detected_distortion: ", state["detected_distortion"])
    distortion = state["detected_distortion"]
    template = retrieve_reframe_template(retriever, distortion) if distortion else ""
    print("template: ", template)
    reply = reframe_chain.invoke({
        "tmpl": template, 
        "u": state["last_user_msg"]
    })
    print("reply: ", reply)
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    print("#########reframe_prompt_node#########")
    return state

def therapy_planner_node(state: ChatState) -> ChatState:
    print("#########therapy_planner_node#########")
    diagnosis: Diagnosis = state.get("diagnosis", "none")
    filters = {"doc_type": "therapy_script"}
    if diagnosis != "none":
        filters["therapy_diagnosis"] = diagnosis
    therapy_script = retrieve_therapy_script(retriever, diagnosis)
    print("therapy_script: ", therapy_script)
    state.update(therapy_script=therapy_script, therapy_attempts=0)  # type: ignore[arg-type]
    print("#########therapy_planner_node#########")
    return state


def guide_exercise_node(state: ChatState) -> ChatState:
    print("#########guide_exercise_node#########")
    script = state.get("therapy_script", "Let's begin a breathing exercise…")
    intro = script.split("\n")[0]
    reply = f"Let's try this together. {intro}\nWhen you're ready, let me know how that felt."
    print("reply: ", reply)
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    state["therapy_attempts"] = state.get("therapy_attempts", 0) + 1  # type: ignore[index]
    print("#########guide_exercise_node#########")
    return state


def collect_feedback_node(state: ChatState) -> ChatState:
    print("#########collect_feedback_node#########")
    user_msg = state["last_user_msg"]
    print("user_msg: ", user_msg)
    sentiment_raw = sentiment_chain.invoke({"feedback": user_msg}).strip().lower()
    sentiment = sentiment_raw if sentiment_raw in {"positive", "neutral", "negative"} else "neutral"
    state.update(feedback_sentiment=sentiment)  # type: ignore[arg-type]
    print("#########collect_feedback_node#########")
    return state


def adjust_instruction_node(state: ChatState) -> ChatState:
    print("#########adjust_instruction_node#########")
    sentiment = state.get("feedback_sentiment", "neutral")
    if sentiment == "negative":
        reply = (
            "Thanks for letting me know. Let's try an alternative approach—perhaps a sensory grounding exercise. "
            "Focus on naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste."
        )
    else:
        reply = "Great job! We'll keep practicing this therapy since it seems helpful."
    state.setdefault("chat_history", []).append(AIMessage(content=reply))
    print("#########adjust_instruction_node#########")
    return state


def summary_writer_node(state: ChatState) -> ChatState:
    print("#########summary_writer_node#########")
    convo = "\n".join(m.content for m in state.get("chat_history", []))
    print("convo: ", convo)
    summary = summary_chain.invoke({"conv": convo})
    print("summary: ", summary)
    # Persist into vector DB
    doc = Document(
        page_content=summary,
        metadata={
            "user_id": state["user_id"],
            "timestamp": str(int(time.time())),
            "doc_type": "session_summary",
        }
    )
    retriever.vectorstore.add_documents([doc])
    logger.info("Session summary stored. Length: %d chars", len(summary))
    print("#########summary_writer_node#########")
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

def needs_therapy_script(state: ChatState) -> bool:
    print("needs_therapy_script: ", state.get("needs_therapy"))
    return state.get("needs_therapy")


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

    sg.add_edge("crisis_path", END)

    # Main flow
    sg.add_edge("session_initializer", "diagnose")

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