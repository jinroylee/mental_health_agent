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
        # Use MMR retriever for better diversity
        docs = retriever.invoke(q, filter=filters)
        runs.append(docs)
    fused = rrf_fuse(runs, k=30, C=60)
    return fused

def top_k_for_generation(retriever, user_message: str, filters: Optional[dict] = None, top_k: int = 5, similarity_threshold: float = 0.25) -> List[Document]:
    """
    Retrieve top-k documents with improved similarity filtering.
    
    Args:
        similarity_threshold: Minimum cosine similarity score (default 0.3, lowered from 0.5)
    """
    fused = retrieve_with_rewrites(retriever, user_message, filters)
    if not fused:
        return []

    # Compute embedding similarity of user query to each doc with moderate threshold
    qvec = embed.embed_query(user_message[:1000])
    
    def cos(a, b):
        num = sum(x*y for x, y in zip(a, b))
        da = math.sqrt(sum(x*x for x in a))
        db = math.sqrt(sum(y*y for y in b))
        return num / (da * db + 1e-9)

    kept = []
    for d in fused:
        # Re-embed the document text head for similarity check
        dvec = embed.embed_query(d.page_content[:512])
        similarity_score = cos(qvec, dvec)
        if similarity_score >= similarity_threshold:
            kept.append(d)
            pretty_logger.info(f"Document kept with similarity: {similarity_score:.3f}")
        else:
            pretty_logger.info(f"Document filtered out with similarity: {similarity_score:.3f}")
        if len(kept) >= top_k:
            break
    
    pretty_logger.state_print("Documents kept after filtering", len(kept))
    return kept

def format_context_documents(docs: List[Document], doc_type: str = "resource") -> str:
    """
    Format retrieved documents into a clear, structured context string.
    """
    if not docs:
        return f"No relevant {doc_type} found in the knowledge base."
    
    if len(docs) == 1:
        return f"Retrieved {doc_type}:\n\n{docs[0].page_content.strip()}"
    
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        formatted_parts.append(f"{doc_type.title()} {i}:\n{doc.page_content.strip()}")
    
    return f"Retrieved {len(docs)} relevant {doc_type}s:\n\n" + "\n\n---\n\n".join(formatted_parts)

def validate_context_quality(context: str, user_query: str) -> dict:
    """
    Validate the quality and relevance of retrieved context.
    
    Returns:
        dict with keys: is_relevant, confidence_score, has_specific_content, recommendations
    """
    # Basic quality checks
    is_fallback = any(phrase in context.lower() for phrase in [
        "no relevant", "no specific", "not found", "error retrieving", "using fallback"
    ])
    
    has_specific_content = len(context.strip()) > 100 and not is_fallback
    
    # Simple relevance check based on keyword overlap
    query_words = set(user_query.lower().split())
    context_words = set(context.lower().split())
    overlap_ratio = len(query_words.intersection(context_words)) / max(len(query_words), 1)
    
    confidence_score = 0.8 if has_specific_content and overlap_ratio > 0.2 else 0.3
    
    recommendations = []
    if is_fallback:
        recommendations.append("Using fallback content - consider expanding knowledge base")
    if overlap_ratio < 0.1:
        recommendations.append("Low keyword overlap - context may not be highly relevant")
    if len(context.strip()) < 50:
        recommendations.append("Very short context - may lack sufficient detail")
    
    return {
        "is_relevant": has_specific_content and overlap_ratio > 0.1,
        "confidence_score": confidence_score,
        "has_specific_content": has_specific_content,
        "is_fallback": is_fallback,
        "recommendations": recommendations
    }

def retrieve_crisis_resource(retriever, locale: str) -> str:
    """Retrieve locale‑specific crisis hotline from the vector store."""
    pretty_logger.logger_separator_top("retrieve_crisis_resource")
    try:
        docs = top_k_for_generation(
            retriever,
            user_message=f"{locale} suicide prevention crisis hotline",
            filters={"doc_type": "crisis_resource"},
            top_k=2,  # Get 2 for redundancy
            similarity_threshold=0.25,  # Lowered threshold for crisis resources (was 0.4)
        )
        
        if docs:
            formatted_context = format_context_documents(docs, "crisis resource")
            pretty_logger.state_print("formatted_context", formatted_context)
            pretty_logger.logger_separator_bottom("retrieve_crisis_resource")
            return formatted_context
        else:
            pretty_logger.warning("No crisis resources found in vector store, using fallback")
            return f"No specific crisis resources found for {locale}. Using fallback:\n\n{CRISIS_RESOURCE_FALLBACK}"
            
    except Exception as exc:
        pretty_logger.warning("Failed to retrieve crisis resource: %s", exc)
        return f"Error retrieving crisis resources. Using fallback:\n\n{CRISIS_RESOURCE_FALLBACK}"

def retrieve_reframe_template(retriever, distortion_label: str) -> str:
    """Retrieve a cognitive distortion reframe template from the vector store."""
    pretty_logger.logger_separator_top("retrieve_reframe_template")
    
    docs = top_k_for_generation(
        retriever,
        user_message=f"Socratic questions cognitive behavioral therapy {distortion_label} reframe",
        filters={"doc_type": "reframe_template"},
        top_k=2,  # Get 2 templates for variety
        similarity_threshold=0.3,  # Lowered threshold (was 0.45)
    )
    
    if docs:
        formatted_context = format_context_documents(docs, "reframing template")
        pretty_logger.state_print("formatted_context", formatted_context)
        pretty_logger.logger_separator_bottom("retrieve_reframe_template")
        return formatted_context
    else:
        fallback = f"No specific reframing template found for '{distortion_label}'. Use these general Socratic questions:\n\n- What evidence supports this thought?\n- What evidence contradicts it?\n- How might someone else view this situation?\n- What would you tell a friend in this situation?\n- What's the most realistic way to look at this?"
        pretty_logger.warning("No reframe templates found, using fallback")
        return fallback

def retrieve_counseling_resource(retriever, query: str) -> str:
    """Retrieve a mental health counseling resource from the vector store."""
    pretty_logger.logger_separator_top("retrieve_counseling_resource")
    
    docs = top_k_for_generation(
        retriever,
        user_message=query,  # Use the original query instead of prefixing
        filters={"doc_type": "counseling_resource"},
        top_k=3,  # Get 3 resources for comprehensive context
        similarity_threshold=0.3,  # Lowered threshold (was 0.5)
    )
    
    if docs:
        formatted_context = format_context_documents(docs, "counseling resource")
        pretty_logger.state_print("formatted_context", formatted_context)
        pretty_logger.logger_separator_bottom("retrieve_counseling_resource")
        return formatted_context
    else:
        fallback = f"No specific counseling resources found for your question. I'll provide general guidance based on evidence-based practices, but you may benefit from consulting with a mental health professional for personalized support."
        pretty_logger.warning("No counseling resources found, using fallback")
        return fallback

def retrieve_therapy_script(retriever, query: str) -> str:
    """Retrieve a coping therapy script matching query (mood/goal)."""
    pretty_logger.logger_separator_top("retrieve_therapy_script")
    
    docs = top_k_for_generation(
        retriever,
        user_message=f"therapy exercise intervention {query}",
        filters={"doc_type": "therapy_resource"},
        top_k=3,  # Get multiple scripts for comprehensive approach
        similarity_threshold=0.3,  # Lowered threshold (was 0.45)
    )
    
    if docs:
        formatted_context = format_context_documents(docs, "therapy script")
        pretty_logger.state_print("formatted_context", formatted_context)
        pretty_logger.logger_separator_bottom("retrieve_therapy_script")
        return formatted_context
    else:
        fallback = "No specific therapy scripts found. Here's a general grounding exercise:\n\nLet's do a simple grounding exercise:\n1. Notice 5 things you can see around you\n2. Notice 4 things you can touch\n3. Notice 3 things you can hear\n4. Notice 2 things you can smell\n5. Notice 1 thing you can taste\n\nThis can help bring you into the present moment when feeling overwhelmed."
        pretty_logger.warning("No therapy scripts found, using fallback")
        return fallback