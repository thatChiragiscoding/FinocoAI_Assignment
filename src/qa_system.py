"""
qa_system.py — Core Q&A pipeline.

Flow:
  1. Retrieve top-5 chunks from ChromaDB.
  2. If max similarity < 0.35 → short-circuit to answerable=false.
  3. Format context + fill prompt template (v1 or v2).
  4. Call LLM provider.
  5. Validate and patch the response JSON.
  6. Return the final answer dict.
"""

import time
from typing import Optional

from src.retriever import retrieve, is_low_similarity, format_context
from src.providers import BaseProvider, get_provider
from src.prompts import SYSTEM_PROMPT, PROMPT_VERSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _not_in_document_response(question: str, provider_name: str, latency_ms: int) -> dict:
    return {
        "question": question,
        "answer": "NOT IN DOCUMENT",
        "answerable": False,
        "confidence": 0.0,
        "citations": [],
        "reasoning": "No relevant content found in the document for this question.",
        "latency_ms": latency_ms,
        "provider": provider_name,
    }


def _validate_and_patch(result: dict, question: str, provider_name: str, latency_ms: int) -> dict:
    """Ensure all required fields exist and have correct types."""
    defaults = {
        "question": question,
        "answer": "NOT IN DOCUMENT",
        "answerable": False,
        "confidence": 0.0,
        "citations": [],
        "reasoning": "",
        "latency_ms": latency_ms,
        "provider": provider_name,
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default

    # Type coercions
    result["answerable"] = bool(result["answerable"])
    result["confidence"] = float(result["confidence"])
    result["latency_ms"] = int(result.get("latency_ms", latency_ms))
    result["provider"] = provider_name

    # If answerable=false, enforce empty citations and standard answer
    if not result["answerable"]:
        result["citations"] = []
        if result["answer"].strip().upper() != "NOT IN DOCUMENT":
            result["answer"] = "NOT IN DOCUMENT"

    # Ensure citations have required fields
    clean_citations = []
    for c in result.get("citations", []):
        if isinstance(c, dict):
            clean_citations.append({
                "section": c.get("section", "Unknown Section"),
                "page": int(c.get("page", 0)),
                "snippet": c.get("snippet", ""),
            })
    result["citations"] = clean_citations

    return result


# ---------------------------------------------------------------------------
# Main Q&A function
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    provider: BaseProvider,
    prompt_version: str = "v2",
    chroma_path: str = "./chroma_db",
) -> dict:
    """
    Run the full RAG → LLM pipeline for a single question.

    Args:
        question       : the question string
        provider       : an instantiated BaseProvider (Gemini or Groq)
        prompt_version : "v1" or "v2"
        chroma_path    : path to the ChromaDB persistent store

    Returns:
        dict matching the required output schema
    """
    t0 = time.time()

    # --- Step 1: Retrieve ---
    chunks, max_score = retrieve(question, chroma_path=chroma_path)
    retrieval_ms = int((time.time() - t0) * 1000)

    # --- Step 2: Low-similarity guard ---
    if is_low_similarity(max_score):
        total_ms = int((time.time() - t0) * 1000)
        result = _not_in_document_response(question, provider.name, total_ms)
        result["reasoning"] = (
            f"Max retrieval similarity {max_score:.3f} is below threshold 0.35 — "
            "question is unlikely to be answerable from the document."
        )
        return result

    # --- Step 3: Build prompt ---
    context = format_context(chunks)
    prompt_template = PROMPT_VERSIONS.get(prompt_version, PROMPT_VERSIONS["v2"])
    user_prompt = prompt_template.format(context=context, question=question)

    # --- Step 4: Call LLM ---
    try:
        result = provider.complete(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        total_ms = int((time.time() - t0) * 1000)
        result = _not_in_document_response(question, provider.name, total_ms)
        result["reasoning"] = f"LLM call failed: {e}"
        return result

    # --- Step 5: Validate + patch ---
    total_ms = int((time.time() - t0) * 1000)
    result = _validate_and_patch(result, question, provider.name, total_ms)

    return result
