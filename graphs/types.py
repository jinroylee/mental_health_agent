""" Types for the mental health assistant.
"""

from typing import List, Literal, Optional, TypedDict
from langchain_core.messages import  BaseMessage

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
    needs_therapy: bool  # branch flag for therapy vs. knowledge

    # Cognitive distortion
    detected_distortion: Optional[str]

    # therapy loop
    therapy_script: Optional[str]
    therapy_attempts: int
    feedback_sentiment: Optional[str]
    is_feedback: bool

    # Summaries
    prior_summary: Optional[str]

    # Last user message (set each turn)
    last_user_msg: str