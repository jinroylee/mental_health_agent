"""
Utility functions for the mental health assistant.
"""

import logging
import os
import math
from typing import Optional, List, Dict

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.documents import Document

from graphs.prompts import CRISIS_RESOURCE_FALLBACK, QUERY_REWRITE_SYSTEM_PROMPT
from graphs.pretty_logging import get_pretty_logger
from graphs.shared_config import llm, embed

logger = logging.getLogger(__name__)
pretty_logger = get_pretty_logger(__name__)
parser = CommaSeparatedListOutputParser()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _similarity_search(retriever, query: str, *, filters: Optional[dict] = None, k: int = 3):
    return retriever.vectorstore.similarity_search(query=query, k=k, filter=filters or {})

def rewrite_queries(user_message: str) -> List[str]:
    text = user_message.strip()
    # Hard truncate very long inputs for embedding health
    if len(text) > 1000:
        text = text[:1000]
    out = llm.invoke(QUERY_REWRITE_SYSTEM_PROMPT.format(user_message=text))
    try:
        variants = parser.parse(out.content)
    except Exception:   
        variants = []
    # Always include the raw query too
    variants = [text] + [v.strip() for v in variants if v.strip()]
    # Deduplicate and cap
    seen = set()
    uniq = []
    for v in variants:
        if v.lower() not in seen:
            uniq.append(v)
            seen.add(v.lower())
    return uniq[:5]  # raw + up to 4 rewrites

def rrf_fuse(results: List[List[Document]], k=60, C=60) -> List[Document]:
    # results: list of lists; each inner list is ranked docs for one query
    # C: constant in RRF; higher -> flatter contribution of lower ranks
    scores: Dict[str, float] = {}
    by_id: Dict[str, Document] = {}
    for run in results:
        for rank, doc in enumerate(run[:k], start=1):
            doc_id = doc.metadata.get("id") or doc.metadata.get("source") or doc.page_content[:64]
            by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (C + rank)
    # sort by fused score
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [by_id[_id] for _id, _ in fused]

def retrieve_with_rewrites(retriever, user_message: str, filters: Optional[dict] = None) -> List[Document]:
    queries = rewrite_queries(user_message)
    runs = []
    for q in queries:
        # Use MMR retriever; you can also call store.max_marginal_relevance_search(...)
        docs = retriever.invoke(q, filter=filters)
        runs.append(docs)
    fused = rrf_fuse(runs, k=30, C=60)
    return fused

def top_k_for_generation(retriever, user_message: str, filters: Optional[dict] = None, top_k: int = 5) -> List[Document]:
    fused = retrieve_with_rewrites(retriever, user_message, filters)
    if not fused:
        return []

    # Optional: compute embedding similarity of user query to each doc to gate low-sim hits
    qvec = embed.embed_query(user_message[:1000])
    # Pinecone returns scores if you call the store directly; here's a simple cosine
    def cos(a, b):
        num = sum(x*y for x, y in zip(a, b))
        da = math.sqrt(sum(x*x for x in a))
        db = math.sqrt(sum(y*y for y in b))
        return num / (da * db + 1e-9)

    kept = []
    for d in fused:
        # If you stored chunk embeddings elsewhere, reuse; otherwise quick re-embed the doc text head
        dvec = embed.embed_query(d.page_content[:512])
        if cos(qvec, dvec) >= 0.25:   # tune threshold on your eval set
            kept.append(d)
        if len(kept) >= top_k:
            break
    pretty_logger.state_print("kept", kept)
    return kept

def retrieve_crisis_resource(retriever, locale: str) -> str:
    """Retrieve locale‑specific crisis hotline from the vector store."""
    pretty_logger.logger_separator_top("retrieve_crisis_resource")
    try:
        docs = top_k_for_generation(
            retriever,
            user_message=f"{locale} suicide prevention",
            filters={"doc_type": "crisis_resource"},
            top_k=1,
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
    docs = top_k_for_generation(
        retriever,
        user_message=f"Socratic questions for {distortion_label}",
        filters={"doc_type": "reframe_template"},
        top_k=1,
    )
    pretty_logger.state_print("docs", docs[0].page_content)
    pretty_logger.logger_separator_bottom("retrieve_reframe_template")
    return docs[0].page_content if docs else "Could there be another perspective on this?"

def retrieve_counseling_resource(retriever, query: str) -> str:
    """Retrieve a mental health counseling resource from the vector store."""
    pretty_logger.logger_separator_top("retrieve_counseling_resource")
    docs = top_k_for_generation(
        retriever,
        user_message=f"Counseling resources for {query}",
        filters={"doc_type": "counseling_resource"},
        top_k=1,
    )
    pretty_logger.state_print("docs", docs[0].page_content)
    pretty_logger.logger_separator_bottom("retrieve_counseling_resource")
    return docs[0].page_content if docs else "Could there be another perspective on this?"

def retrieve_therapy_script(retriever, query: str) -> str:
    """Retrieve a coping therapy script matching query (mood/goal)."""
    pretty_logger.logger_separator_top("retrieve_therapy_script")
    docs = top_k_for_generation(
        retriever,
        user_message=query,
        filters={"doc_type": "therapy_resource"},
        top_k=3,
    )
    script = "\n\n".join(d.page_content for d in docs) or "Let's do a simple grounding exercise: notice 5 things you can see…"
    pretty_logger.state_print("script", script)
    pretty_logger.logger_separator_bottom("retrieve_therapy_script")
    return script