"""
Utility functions for the mental health assistant.
"""

import logging
from typing import Optional

from graphs.prompts import CRISIS_RESOURCE_FALLBACK

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _similarity_search(retriever, query: str, *, filters: Optional[dict] = None, k: int = 3):
    return retriever.vectorstore.similarity_search(query=query, k=k, filter=filters or {})

def retrieve_crisis_resource(retriever, locale: str) -> str:
    """Retrieve locale‑specific crisis hotline from the vector store."""
    logger.info("==========retrieve_crisis_resource ==================")
    try:
        docs = retriever.vectorstore.similarity_search(
            query=f"{locale} suicide prevention", k=1, filter={"doc_type": "crisis_resource"}
        )
        print("docs: ", docs)

        logger.info("==========retrieve_crisis_resource ==================")
        return docs[0].page_content if docs else CRISIS_RESOURCE_FALLBACK
    except Exception as exc:
        logger.warning("Failed to retrieve crisis resource: %s", exc)
        return CRISIS_RESOURCE_FALLBACK

def retrieve_reframe_template(retriever, distortion_label: str) -> str:
    """Retrieve a cognitive distortion reframe template from the vector store."""
    logger.info("==========retrieve_reframe_template ==================")
    docs = _similarity_search(retriever,
        query=f"Socratic questions for {distortion_label}",
        filters={"doc_type": "reframe_template", "distortion_label": distortion_label},
        k=1,
    )
    logger.info("==========retrieve_reframe_template ==================")
    return docs[0].page_content if docs else "Could there be another perspective on this?"

def retrieve_therapy_script(retriever, query: str) -> str:
    """Retrieve a coping therapy script matching query (mood/goal)."""
    logger.info("==========retrieve_therapy_script ==================")
    docs = retriever.vectorstore.similarity_search(
        query=query,
        k=1,
        filter={"doc_type": "therapy_script"},
    )
    logger.info("==========retrieve_therapy_script ==================")
    return docs[0].page_content if docs else "Let's practice slow, deep breathing together."  