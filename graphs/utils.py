"""
Utility functions for the mental health assistant.
"""

import logging
from typing import Optional

from graphs.prompts import CRISIS_RESOURCE_FALLBACK
from graphs.pretty_logging import get_pretty_logger

logger = logging.getLogger(__name__)
pretty_logger = get_pretty_logger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _similarity_search(retriever, query: str, *, filters: Optional[dict] = None, k: int = 3):
    return retriever.vectorstore.similarity_search(query=query, k=k, filter=filters or {})

def retrieve_crisis_resource(retriever, locale: str) -> str:
    """Retrieve locale‑specific crisis hotline from the vector store."""
    pretty_logger.logger_separator_top("retrieve_crisis_resource")
    try:
        docs = _similarity_search(
            retriever,
            query=f"{locale} suicide prevention",
            filters={"doc_type": "crisis_resource"},
            k=1,
        )
        pretty_logger.state_print("docs", docs[0].page_content)

        pretty_logger.logger_separator_bottom("retrieve_crisis_resource")
        return docs[0].page_content if docs else CRISIS_RESOURCE_FALLBACK
    except Exception as exc:
        pretty_logger.warning("Failed to retrieve crisis resource: %s", exc)
        return CRISIS_RESOURCE_FALLBACK

def retrieve_reframe_template(retriever, distortion_label: str) -> str:
    """Retrieve a cognitive distortion reframe template from the vector store."""
    pretty_logger.logger_separator_top("retrieve_reframe_template")
    docs = _similarity_search(retriever,
        query=f"Socratic questions for {distortion_label}",
        filters={"doc_type": "reframe_template"},
        k=1,
    )
    pretty_logger.state_print("docs", docs[0].page_content)
    pretty_logger.logger_separator_bottom("retrieve_reframe_template")
    return docs[0].page_content if docs else "Could there be another perspective on this?"

def retrieve_therapy_script(retriever, query: str) -> str:
    """Retrieve a coping therapy script matching query (mood/goal)."""
    pretty_logger.logger_separator_top("retrieve_therapy_script")
    docs = _similarity_search(
        retriever,
        query=query,
        filters={"doc_type": "therapy_resource"},
        k=3,
    )
    script = "\n\n".join(d.page_content for d in docs) or "Let's do a simple grounding exercise: notice 5 things you can see…"
    pretty_logger.state_print("script", script)
    pretty_logger.logger_separator_bottom("retrieve_therapy_script")
    return script