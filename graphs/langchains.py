"""
Langchain definitions for the mental health assistant.
"""
import logging
from pathlib import Path
import dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptValue
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from graphs.pretty_logging import get_pretty_logger
from graphs.prompts import (
    FEEDBACK_CLASSIFICATION_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    DISTORTION_SYSTEM_PROMPT,
    CRISIS_SYSTEM_PROMPT,
    ADAPTIVE_COUNSELING_PROMPT,
    REFRAME_SYSTEM_PROMPT,
    GUIDE_EXERCISE_SYSTEM_PROMPT,
    ADJUST_INSTRUCTION_SYSTEM_PROMPT,
)
from graphs.shared_config import llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pretty_logger = get_pretty_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

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
    ("system", "Output exactly one of: happy|sad|anxious|angry|neutral|stressed"),
    ("user", "{text}"),
])
mood_chain = mood_prompt | llm | StrOutputParser()

classify_feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", FEEDBACK_CLASSIFICATION_SYSTEM_PROMPT),
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
    ("system", "Output exactly one of: positive|neutral|negative"),
    ("user", "{feedback}"),
])
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# Crisis response chain - enhanced with better context formatting
crisis_prompt = ChatPromptTemplate.from_messages([
    ("system", CRISIS_SYSTEM_PROMPT),
    ("system", "=== CRISIS RESOURCES ===\n{resources}\n=== END CRISIS RESOURCES ==="),
    ("user", "{user_message}"),
])
crisis_chain = crisis_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Counseling dialogue chain - completely revamped with specialized prompt
counseling_prompt = ChatPromptTemplate.from_messages([
    ("system", ADAPTIVE_COUNSELING_PROMPT),
    ("system", "=== SESSION CONTEXT ===\nPrior conversation summary: {prior_summary}\n=== END SESSION CONTEXT ==="),
    ("system", "=== RETRIEVED COUNSELING KNOWLEDGE ===\n{ctx}\n=== END RETRIEVED KNOWLEDGE ===\n\nUse the above knowledge as your primary reference for responding to the user's question."),
    MessagesPlaceholder("history"),
    ("user", "{question}"),
])
counseling_chain = counseling_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Reframe response chain - updated with specialized prompt
reframe_prompt = ChatPromptTemplate.from_messages([
    ("system", REFRAME_SYSTEM_PROMPT),
    ("system", "=== SESSION CONTEXT ===\nPrior conversation summary: {prior_summary}\n=== END SESSION CONTEXT ==="),
    ("system", "=== SOCRATIC QUESTIONING TEMPLATE ===\n{tmpl}\n=== END TEMPLATE ===\n\nUse the above template to guide your Socratic questioning approach."),
    ("user", "User said: {u}\n\nPlease respond using Socratic coaching techniques to help them explore their thoughts."),
])
reframe_chain = reframe_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Guide exercise chain - updated with specialized prompt
guide_exercise_prompt = ChatPromptTemplate.from_messages([
    ("system", GUIDE_EXERCISE_SYSTEM_PROMPT),
    ("system", "=== SESSION CONTEXT ===\nPrior conversation summary: {prior_summary}\n=== END SESSION CONTEXT ==="),
    ("system", "=== THERAPEUTIC EXERCISE SCRIPT ===\n{script}\n=== END SCRIPT ===\n\nUse the above script to guide the user through this therapeutic exercise."),
    ("user", "User said: {u}\n\nPlease provide therapy instructions based on the script above."),
])
guide_exercise_chain = guide_exercise_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Adjust instruction chain - updated with specialized prompt
adjust_instruction_prompt = ChatPromptTemplate.from_messages([
    ("system", ADJUST_INSTRUCTION_SYSTEM_PROMPT),
    ("user", "User feedback about the previous exercise: {u}\n\nPlease adjust the therapeutic approach based on this feedback."),
])
adjust_instruction_chain = adjust_instruction_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Summary generation chain
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the conversation in exactly 3 concise sentences. No bullets, headings, or preamble."),
    ("system", "{conv}"),
])
summary_chain = summary_prompt | llm | StrOutputParser()