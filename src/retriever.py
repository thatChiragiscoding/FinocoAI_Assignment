"""
retriever.py — Query ChromaDB and return top-k chunks with similarity scores.

Key design decision:
- If the max cosine similarity score across top-5 results is < 0.35,
  the question is almost certainly unanswerable from the document.
  The retriever flags this so qa_system.py can force answerable=false.
"""

from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from src.ingest import get_collection, CHROMA_PATH

_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

TOP_K = 10
LOW_SIMILARITY_THRESHOLD = 0.25 #below this → force refusal

# NOTE: all-MiniLM-L6-v2 produces low cosine similarities (0.15–0.40) on
# academic PDFs for asymmetric QA (question vs. document statement).
# Adversarial refusal for q8-q10 is handled by V2 prompt rules, not this threshold.

def retrieve(
    question: str,
    top_k: int = TOP_K,
    chroma_path: str = CHROMA_PATH,
) -> Tuple[List[Dict], float]:
    """
    Embed `question` and return the top-k most similar chunks.

    Returns:
        chunks   : list of dicts with keys {text, page, section, score}
        max_score: highest similarity among the top-k results
    """
    _, collection = get_collection(chroma_path)

    # Query expansion for interpretation questions — boosts retrieval
    # when semantic gap between question phrasing and document text is large
    query = question
    if any(w in question.lower() for w in ["why", "how does", "what is the difference", "distinguish"]):
        query = question + " planning subgoals decomposition question developing queries pipeline stages separate"

    query_emb = _model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # multi-qa-mpnet-base-dot-v1 uses dot product (inner product).
    # With hnsw:space=ip, ChromaDB returns negative inner product as distance.
    # Convert back to actual similarity: similarity = -distance
    chunks = []
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = -dist  # ip space: distance = -dot_product
        chunks.append({
            "text": doc,
            "page": meta.get("page", 0),
            "section": meta.get("section", "Unknown Section"),
            "score": round(similarity, 4),
        })

    max_score = max((c["score"] for c in chunks), default=0.0)
    return chunks, max_score


def is_low_similarity(max_score: float) -> bool:
    """Return True if similarity is too low to trust any answer."""
    return max_score < LOW_SIMILARITY_THRESHOLD


def format_context(chunks: List[Dict]) -> str:
    chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)  # best first
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | Section: {chunk['section']} | Page: {chunk['page']} | Score: {chunk['score']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
