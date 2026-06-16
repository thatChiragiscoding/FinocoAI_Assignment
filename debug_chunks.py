"""Debug: find which chunk contains WENLIN ZHANG and what its similarity is."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.ingest import get_collection

_, col = get_collection()

# Find the chunk(s) containing the first author name
all_results = col.get(include=["documents", "metadatas"])
print("=== Chunks containing 'WENLIN' or 'Wenlin' ===")
for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
    safe = doc.encode('ascii', errors='replace').decode('ascii')
    if 'WENLIN' in doc.upper() or 'WENLIN' in safe.upper():
        print(f"  page={meta['page']} | section={meta['section']}")
        print(f"  {repr(safe[:200])}")
        print()

# Also check page=1 chunks
print("=== All page=1 chunks ===")
for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
    if meta['page'] == 1:
        safe = doc.encode('ascii', errors='replace').decode('ascii')
        print(f"  page=1 | section={meta['section']}")
        print(f"  {repr(safe[:200])}")
        print()

# Raw similarity for different question phrasings
questions = [
    "Who is the first author of this paper?",
    "What is the name of the first author?",
    "author name affiliation institution",
    "WENLIN ZHANG City University Hong Kong",
]
raw_q = col.query(
    query_texts=questions,
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
for q, docs, metas, dists in zip(questions, raw_q["documents"], raw_q["metadatas"], raw_q["distances"]):
    print(f"\nQUERY: {q}")
    for doc, meta, dist in zip(docs, metas, dists):
        sim = round(1.0 - dist, 4)
        safe = doc[:80].encode('ascii', errors='replace').decode('ascii')
        print(f"  sim={sim:.4f} | p={meta['page']} | {repr(safe)}")
