"""
Langchain definitions for the mental health assistant.
"""
import os
import logging
from pathlib import Path
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptValue
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from graphs.pretty_logging import get_pretty_logger
from graphs.prompts import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pretty_logger = get_pretty_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.3)

# ---------------------------------------------------------------------------
# Chain definitions
# ---------------------------------------------------------------------------
def log_llm_input(messages: ChatPromptValue):
    """Log the formatted prompt that gets sent to the LLM."""
    pretty_logger.function_separator("FORMATTED PROMPT TO LLM")
    for msg in messages.to_messages():
        pretty_logger.state_print("Role", msg.type)
        pretty_logger.content_block(str(msg.content), "Content")
        pretty_logger.separator_line("─", 30)
    pretty_logger.function_separator("END FORMATTED PROMPT")
    return messages

def log_llm_output(output):
    """Log the raw LLM output before parsing."""
    pretty_logger.info("Raw LLM output for diagnosis: %s", output.content)
    pretty_logger.function_separator("RAW LLM OUTPUT")
    pretty_logger.content_block(str(output.content))
    pretty_logger.function_separator("END RAW OUTPUT")
    return output

# Mood detection chain
mood_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return one word: happy, sad, anxious, angry, neutral, or stressed."),
    ("user", "{text}"),
])
mood_chain = mood_prompt | llm | StrOutputParser()

classify_feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", "Based on the conversation history, classify if the user's message is a feedback or not. If feedback, return 'feedback'. If not, return 'none'."),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
])
classify_feedback_chain = classify_feedback_prompt | llm | StrOutputParser()

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
    ("system", DISTORTION_SYSTEM_PROMPT),
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
    MessagesPlaceholder("history"),
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

guide_exercise_prompt = ChatPromptTemplate.from_messages([
    ("system", GUIDE_EXERCISE_SYSTEM_PROMPT),
    ("system", "Script:\n{script}"),
    ("user", "User said: {u}\nPlease provide therapy instructions."),
])
guide_exercise_chain = guide_exercise_prompt | llm | StrOutputParser()

adjust_instruction_prompt = ChatPromptTemplate.from_messages([
    ("system", ADJUST_INSTRUCTION_SYSTEM_PROMPT),
    ("user", "{u}"),
])
adjust_instruction_chain = adjust_instruction_prompt | llm | StrOutputParser()

# Summary generation chain
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the key points and progress of the following conversation in 3 sentences."),
    ("system", "{conv}"),
])
summary_chain = summary_prompt | llm | StrOutputParser()