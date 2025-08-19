"""
Langchain definitions for the mental health assistant.
"""
import logging
from pathlib import Path
import dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptValue
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from graphs.pretty_logging import get_pretty_logger
from graphs.prompts import (
    BASE_SYSTEM_PROMPT,
    FEEDBACK_CLASSIFICATION_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    DISTORTION_SYSTEM_PROMPT,
    CRISIS_SYSTEM_PROMPT,
    COUNSELING_SYSTEM_PROMPT,
    REFRAME_SYSTEM_PROMPT,
    GUIDE_EXERCISE_SYSTEM_PROMPT,
    ADJUST_INSTRUCTION_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
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
    MessagesPlaceholder("history", optional=True),
    ("user", "{input}"),
])
_feedback_parser_with_fallback = JsonOutputParser().with_fallbacks([
    RunnableLambda(lambda _: {"is_feedback": False})
])
classify_feedback_chain = classify_feedback_prompt | log_llm_input | llm | log_llm_output | _feedback_parser_with_fallback

# Diagnosis chain with JSON output parser
diagnosis_prompt = ChatPromptTemplate.from_messages([
   ( "system", DIAGNOSIS_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
])

_diagnosis_parser_with_fallback = JsonOutputParser().with_fallbacks([
    RunnableLambda(lambda _: {"needs_therapy": False, "diagnosis": "none"})
])
diagnosis_chain = diagnosis_prompt | llm | _diagnosis_parser_with_fallback

# Distortion detection chain
distortion_prompt = ChatPromptTemplate.from_messages([
    ("system", DISTORTION_SYSTEM_PROMPT),
    ("human", "{message}"),
])
_distortion_parser_with_fallback = JsonOutputParser().with_fallbacks([
    RunnableLambda(lambda _: {"distortion": False, "label": "none"})
])
distortion_chain = distortion_prompt | log_llm_input | llm | log_llm_output | _distortion_parser_with_fallback

# Sentiment analysis chain
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", SENTIMENT_SYSTEM_PROMPT),
    ("user", "{feedback}"),
])
sentiment_chain = sentiment_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Crisis response chain - enhanced with better context formatting
crisis_prompt = ChatPromptTemplate.from_messages([
    ("system", CRISIS_SYSTEM_PROMPT),
    ("system", "=== CRISIS RESOURCES ===\n{resources}\n=== END CRISIS RESOURCES ==="),
    ("user", "{user_message}"),
])
crisis_chain = crisis_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Counseling dialogue chain - completely revamped with specialized prompt
counseling_prompt = ChatPromptTemplate.from_messages([
    ("system", COUNSELING_SYSTEM_PROMPT),
    ("system", "=== RETRIEVED COUNSELING KNOWLEDGE ===\n{ctx}\n=== END RETRIEVED KNOWLEDGE ===\n"),
    ("system", "=== SESSION CONTEXT ===\nPrior conversation summary: {prior_summary}\n=== END SESSION CONTEXT ==="),
    MessagesPlaceholder("history"),
    ("user", "{question}"),
])
counseling_chain = counseling_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Reframe response chain - updated with specialized prompt
reframe_prompt = ChatPromptTemplate.from_messages([
    ("system", REFRAME_SYSTEM_PROMPT),
    ("system", "=== SOCRATIC QUESTIONING TEMPLATE ===\n{tmpl}\n=== END TEMPLATE ===\n"),
    ("system", "=== SESSION CONTEXT ===\nPrior conversation summary: {prior_summary}\n=== END SESSION CONTEXT ==="),
    MessagesPlaceholder("history"),
    ("user", "User said: {u}\n\nPlease respond using Socratic coaching techniques to help them explore their thoughts."),
])
reframe_chain = reframe_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Guide exercise chain - updated with specialized prompt
guide_exercise_prompt = ChatPromptTemplate.from_messages([
    ("system", GUIDE_EXERCISE_SYSTEM_PROMPT),
    ("system", "=== THERAPEUTIC EXERCISE SCRIPT ===\n{script}\n=== END SCRIPT ===\n"),
    ("system", "=== SESSION CONTEXT ===\nPrior conversation summary: {prior_summary}\n=== END SESSION CONTEXT ==="),
    MessagesPlaceholder("history"),
    ("user", "User said: {u}\n\nPlease provide therapy instructions based on the script above."),
])
guide_exercise_chain = guide_exercise_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

wrap_up_prompt = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("user", "{u}"),
])
wrap_up_chain = wrap_up_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Adjust instruction chain - updated with specialized prompt
adjust_instruction_prompt = ChatPromptTemplate.from_messages([
    ("system", ADJUST_INSTRUCTION_SYSTEM_PROMPT),
    ("system", "=== THERAPEUTIC EXERCISE SCRIPT ===\n{script}\n=== END SCRIPT ===\n"),
    MessagesPlaceholder("history"),
    ("user", "{u}"),
])
adjust_instruction_chain = adjust_instruction_prompt | log_llm_input | llm | log_llm_output | StrOutputParser()

# Summary generation chain
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the conversation in exactly 3 concise sentences. No bullets, headings, or preamble."),
    ("system", "{conv}"),
])
summary_chain = summary_prompt | llm | StrOutputParser()