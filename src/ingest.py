import fitz
import re
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"

def get_collection(db_path: str = CHROMA_PATH):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("paper_chunks")  # metadata not needed here
    return client, collection


# ── Hardcoded section map for this specific paper ─────────────────────────────
SECTION_MAP = {
    ("1", "Introduction"): "1 Introduction",
    ("2", "Planning"): "2 Planning",
    ("2.1", "Planning with Structured World Knowledge"): "2.1 Planning with Structured World Knowledge",
    ("2.2", "Planning as a Learnable Process"): "2.2 Planning as a Learnable Process",
    ("2.3", "Discussion"): "2.3 Discussion",
    ("3", "Question Developing"): "3 Question Developing",
    ("3.1", "Definition"): "3.1 Definition",
    ("3.2", "Reward-Optimized Methods"): "3.2 Reward-Optimized Methods",
    ("3.3", "Supervision-Driven Methods"): "3.3 Supervision-Driven Methods",
    ("3.4", "Discussion"): "3.4 Discussion",
    ("4", "Web Exploration"): "4 Web Exploration",
    ("4.1", "Web retrieval agents"): "4.1 Web retrieval agents",
    ("4.2", "API-Based Retrieval Systems"): "4.2 API-Based Retrieval Systems",
    ("4.3", "Discussion"): "4.3 Discussion",
    ("5", "Report Generation"): "5 Report Generation",
    ("5.1", "Structure Control"): "5.1 Structure Control",
    ("5.2", "Factual Integrity"): "5.2 Factual Integrity",
    ("5.3", "Discussion"): "5.3 Discussion",
    ("6", "Optimization"): "6 Optimization",
    ("6.1", "Workflow"): "6.1 Workflow",
    ("6.2", "Parameter Optimization"): "6.2 Parameter Optimization",
    ("7", "Benchmark and Evaluation"): "7 Benchmark and Evaluation",
    ("8", "Limitations and Future Directions"): "8 Limitations and Future Directions",
    ("9", "Conclusion"): "9 Conclusion",
}

# Lines to skip — noise from PDF headers/footers
SKIP_LINES = {
    "",
    "Manuscript submitted to ACM",
    "Wenlin Zhang et al.",
    "Deep Research: A Survey of Autonomous Research Agents",
    "arXiv:2508.12752v1  [cs.IR]  18 Aug 2025",
}

NUMBER_PATTERN = re.compile(r'^\d+(\.\d+)?$')  # matches "1", "2.1", "4.3" etc


def clean_text(text: str) -> str:
    # Extended range to cover Mathematical Alphanumeric Symbols block
    text = re.sub(r'[\u1d00-\u1d7ff\u2000-\u2fff\u3000-\uffff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def should_skip(line: str) -> bool:
    line = line.strip()
    if line in SKIP_LINES:
        return True
    return False


def extract_full_text_with_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def build_section_chunks(pages: list[dict]) -> list[dict]:
    """
    Parse all lines across all pages.
    Detect section headers by looking at TWO consecutive lines:
      line N   = section number  e.g. "2.1"
      line N+1 = section title   e.g. "Planning with Structured World Knowledge"
    """
    all_lines = []
    for page_data in pages:
        lines = page_data["text"].split('\n')
        for line in lines:
            all_lines.append({
                "text": line.strip(),
                "page": page_data["page"]
            })

    chunks = []
    current_section = "Abstract"
    current_text = []
    current_page = 1

    i = 0
    while i < len(all_lines):
        line = all_lines[i]["text"]
        page = all_lines[i]["page"]

        # ── Try to detect a two-line section header ───────────────────────────
        if NUMBER_PATTERN.match(line) and i + 1 < len(all_lines):
            next_line = all_lines[i + 1]["text"]
            key = (line, next_line)

            if key in SECTION_MAP:
                # Save current section
                text_so_far = " ".join(current_text).strip()
                if text_so_far and len(text_so_far) > 30:
                    chunks.append({
                        "section": current_section,
                        "text": text_so_far,
                        "start_page": current_page,
                        "end_page": page,
                    })

                # Start new section
                current_section = SECTION_MAP[key]
                current_text = []
                current_page = page
                i += 2  # skip both the number line and title line
                continue

        # ── Skip noise lines ──────────────────────────────────────────────────
        if line in SKIP_LINES:
            i += 1
            continue

        # ── Skip standalone page numbers ──────────────────────────────────────
        if re.match(r'^\d{1,2}$', line):
            if i + 1 < len(all_lines):
                next_line = all_lines[i + 1]["text"]
                if (line, next_line) not in SECTION_MAP:
                    i += 1
                    continue
            else:
                i += 1
                continue

        # ── Regular content line ──────────────────────────────────────────────
        if line:
            current_text.append(line)

        i += 1

    # Save the last section
    text_so_far = " ".join(current_text).strip()
    if text_so_far and len(text_so_far) > 30:
        chunks.append({
            "section": current_section,
            "text": text_so_far,
            "start_page": current_page,
            "end_page": pages[-1]["page"],
        })

    return chunks


def split_large_sections(
    chunks: list[dict],
    max_words: int = 500
) -> list[dict]:
    """Split sections longer than max_words into overlapping sub-chunks."""
    final_chunks = []
    overlap = 80

    for chunk in chunks:
        words = chunk["text"].split()
        if len(words) <= max_words:
            final_chunks.append(chunk)
        else:
            start = 0
            sub_idx = 1
            while start < len(words):
                end = min(start + max_words, len(words))
                sub_text = " ".join(words[start:end])
                final_chunks.append({
                    "section": chunk["section"],
                    "text": sub_text,
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                    "sub_chunk": sub_idx,
                })
                sub_idx += 1
                if end == len(words):
                    break
                start += max_words - overlap

    return final_chunks


def build_index(
    pdf_path: str = "paper/deep_research_survey.pdf",
    db_path: str = "./chroma_db"
):
    print(f"Reading PDF: {pdf_path}")
    pages = extract_full_text_with_pages(pdf_path)
    print(f"   Found {len(pages)} pages")

    print("Building section chunks...")
    section_chunks = build_section_chunks(pages)
    print(f"   Found {len(section_chunks)} sections:")
    for c in section_chunks:
        word_count = len(c["text"].split())
        print(f"   [{c['start_page']:2d}p] {c['section']:<55} ({word_count} words)")

    print("\nSplitting large sections...")
    chunks = split_large_sections(section_chunks, max_words=500)
    print(f"   Total chunks after splitting: {len(chunks)}")

    # FIX: Clean all chunk texts before embedding to remove unicode math
    # symbols that crash Groq's JSON validator
    print("\nCleaning text (stripping unicode math symbols)...")
    for chunk in chunks:
        chunk["text"] = clean_text(chunk["text"])
    print("   Done cleaning.")

    print("\nEmbedding chunks...")
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection("paper_chunks")
        print("   Deleted old collection")
    except Exception:
        pass

    collection = client.create_collection("paper_chunks", metadata={"hnsw:space": "ip"})

    ids, metadatas = [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i:04d}")
        metadatas.append({
            "section": chunk["section"],
            "page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "sub_chunk": chunk.get("sub_chunk", 1),
        })

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\nDone! Indexed {len(chunks)} chunks into ChromaDB")
    print(f"   DB saved at: {db_path}")
    return collection


def ingest(pdf_path: str = "paper/deep_research_survey.pdf") -> int:
    collection = build_index(pdf_path)
    return collection.count()


if __name__ == "__main__":
    build_index()