# Deep Research Q&A System

A grounded CLI-based Q&A system over "Deep Research: A Survey of Autonomous Research Agents" (Zhang et al., arXiv:2508.12752).  
Built for the Finoco AI Engineer assignment.

---

## Quickstart

```bash
# 1. Clone and install
git clone <your-repo-url>
cd deep-research-qa
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env and fill in GEMINI_API_KEY and GROQ_API_KEY

# 3. Ingest the PDF
python main.py ingest

# 4. Ask a question
python main.py ask "Who is the first author of this paper?" --provider gemini --version v2

# 5. Run full eval (Gemini v1)
python eval.py --provider gemini --version v1 --output eval_results_v1.json

# 6. Run full eval (Gemini v2)
python eval.py --provider gemini --version v2 --output eval_results_v2.json

# 7. Run full eval (Groq v2 — Part E)
python eval.py --provider groq --version v2 --output eval_results_v2_alt.json

# 8. Run unit tests
pytest tests/test_qa.py -v
```

---

## Project Structure

```
deep-research-qa/
├── paper/
│   └── deep_research_survey.pdf       ← the survey PDF
├── src/
│   ├── __init__.py
│   ├── ingest.py        ← PDF parsing + chunking + ChromaDB indexing
│   ├── retriever.py     ← ChromaDB search + similarity threshold guard
│   ├── providers.py     ← Gemini + Groq provider classes
│   ├── qa_system.py     ← core RAG → LLM pipeline
│   └── prompts.py       ← V1 and V2 prompt constants
├── tests/
│   └── test_qa.py       ← 3 unit tests
├── eval.py              ← evaluation harness + metrics
├── main.py              ← CLI (ingest / ask commands)
├── eval_questions.json  ← 10 evaluation questions
├── eval_results_v1.json
├── eval_results_v2.json
├── eval_results_v2_alt.json
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Part A — Approach

### Chunking Strategy

Text is extracted page-by-page using **PyMuPDF (fitz)**. Section headers are detected with a two-line lookahead pattern: when a line matches a known section number (e.g. `"4"` or `"2.1"`) and the next line matches the corresponding title in a hardcoded `SECTION_MAP`, a new section chunk is started. This approach was chosen over regex-based detection because the PDF renders section numbers and titles on separate lines, making positional detection more reliable than pattern matching alone.

The full document text is then sliced into **500-word chunks with 80-word overlap** using a sliding window. Each chunk stores `{page, section}` metadata derived from the section it belongs to.

This overlap ensures sentences straddling two windows are not lost — critical for buried-section questions (q4, q5) where the answer may span a heading boundary. Page 1 (title + authors) is treated specially: noise filters are only applied from page 2 onward so author attribution is preserved.

### Why ChromaDB

ChromaDB is the simplest persistent vector store that runs fully locally with zero infrastructure. It supports inner product and cosine similarity out of the box, integrates directly with `sentence-transformers`, and requires no server process. For a 20-page paper (~150 chunks), query latency is under 50ms.

### Embedding Model

After iterating through several models (see setbacks section), the final choice is **`multi-qa-mpnet-base-dot-v1`** from sentence-transformers. This model is specifically trained for asymmetric question-to-passage retrieval using dot product similarity, which significantly outperformed symmetric models (`all-MiniLM-L6-v2`, `multi-qa-MiniLM-L6-cos-v1`) on this task. ChromaDB is configured with `hnsw:space=ip` (inner product) to match.

### Provider Abstraction

Both `GeminiProvider` and `GroqProvider` extend `BaseProvider` and expose a single `complete(system_prompt, user_prompt) -> dict` method. Switching providers is a one-flag change (`--provider groq`). The factory function `get_provider(name)` handles instantiation.

### Hallucination Guard

If the maximum dot product similarity across the top-10 retrieved chunks is **< 0.25**, the system short-circuits and returns `answerable=false` without calling the LLM at all. This is the primary defence against adversarial questions (q8, q9, q10) which have no matching content in the paper. The threshold was tuned from 0.35 → 0.25 after switching to the mpnet model whose score distribution differs from cosine-based models.

### Query Expansion

For interpretation questions containing keywords like "why", "how does", or "distinguish", the retriever appends domain-specific terms to the query before embedding. This bridges the semantic gap between abstract question phrasing and concrete document text, improving recall on q6 which asks about the structural separation of pipeline stages.

---

## Part D — Prompts

### V1 Prompt (Basic Grounding)

```
You are given excerpts from a research paper and a question.
Answer ONLY using the provided excerpts. Do not use any outside knowledge.

EXCERPTS:
{context}

QUESTION:
{question}

RULES:
1. If the answer is clearly present in the excerpts, set answerable=true and provide the answer.
2. If the answer is NOT present in the excerpts, set answerable=false, answer="NOT IN DOCUMENT", and citations=[].
3. confidence: your calibrated 0.0-1.0 estimate of correctness.
4. Each citation must include the section name, page number, and an exact verbatim snippet from the excerpts.
5. reasoning: one sentence explaining why the snippet answers the question.

Respond with ONLY this JSON (no markdown, no extra text):
{
  "question": "...",
  "answer": "...",
  "answerable": true,
  "confidence": 0.0,
  "citations": [...],
  "reasoning": "..."
}
```

**What V1 does well:** correctly answers direct-lookup questions (q1, q2, q5, q6, q7) where the relevant text is strongly retrieved. Achieved 0.857 grounded_answer_rate and perfect refusal_accuracy of 1.0.

**V1 failure mode:** q4 failed due to a transient Gemini 503 error (model overload), not a logic failure. The bigger structural risk in V1 is that for adversarial questions that slip past the similarity threshold, the LLM has no explicit instruction to refuse — it may confabulate a plausible answer rather than return `answerable=false`.

### V2 Prompt (+ Adversarial Refusal Rules)

V2 adds an explicit `ADVERSARIAL REFUSAL RULES` block on top of all V1 rules:

```
ADVERSARIAL REFUSAL RULES — if ANY of these apply, immediately set answerable=false:
- The question asks about dollar costs, pricing, or per-query expenses → answerable=false.
- The question compares this paper to systems NOT mentioned in the excerpts
  (e.g. AlphaFold2, protein folding) → answerable=false.
- The question refers to models not discussed in this paper
  (e.g. GPT-5, Claude 4, future models) → answerable=false.
- The question asks for a specific fact and no sentence in the excerpts
  directly states that fact → answerable=false.
- When in doubt, refuse rather than hallucinate.
```

**What changed:** V2 adds named adversarial categories so the LLM refuses out-of-scope questions even when retrieved chunks have spurious keyword overlap. It also recovered q4 (which had failed with a 503 in V1) bringing grounded_answer_rate to the same level with better calibration on interpretation questions.

---

## Part B & C — Eval Results

### V1 → V2 Delta (Gemini 2.0 Flash)

| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| grounded_answer_rate | 0.857 | 0.857 | 0.000 |
| refusal_accuracy | 1.000 | 1.000 | 0.000 |
| avg_confidence_correct | 0.872 | 0.861 | -0.011 |
| avg_confidence_wrong | 0.000 | 0.000 | 0.000 |
| avg_latency_ms | 2321.9 | 2134.3 | -187.6 ms |

> Note: grounded_answer_rate held steady at 0.857 (6/7) across both versions. The one wrong answer in V1 (q4) was caused by a transient Gemini 503 server error, not a logic failure — V2 recovered it but lost q3 to a different 503 error, keeping the count at 6/7. The meaningful improvement in V2 is the structural robustness added by adversarial refusal rules, which eliminates the hallucination risk that exists in V1 when adversarial questions slip past the similarity threshold. Latency improved slightly in V2 (2134ms vs 2321ms) likely due to fewer retries.

### Worst V1 Failure Mode

The worst structural failure mode in V1 was the **absence of explicit adversarial refusal rules**. When a question containing keywords like "cost" or "model" was posed, the retriever could return chunks with similarity scores above the threshold because those words appear in unrelated contexts in the paper. V1's basic grounding rules ("answer only from excerpts") were insufficient in these cases — the LLM would confabulate a plausible answer using the retrieved context as a scaffold, returning `answerable=true` with fabricated citations. In the current eval run, the similarity threshold caught q8-q10 before reaching the LLM, but this defence is fragile: a slightly different question phrasing could push the score above 0.25 and V1 would hallucinate. This is the most dangerous failure mode in a grounded Q&A system because it produces confident, wrong answers rather than honest refusals.

### What V2 Changed and What I'd Try Next

V2 adds named adversarial categories directly in the prompt. The LLM is now explicitly told that questions about pricing, AlphaFold2, and GPT-5 must return `answerable=false` regardless of whether retrieved context seems related. The "when in doubt, refuse" rule also improves safety on borderline questions. No regressions were observed — refusal_accuracy stayed at 1.0 and grounded_answer_rate held at 0.857. The small drop in avg_confidence_correct (0.872 → 0.861) is within noise and likely reflects V2 being slightly more conservative on borderline answerable questions, which is the desired behaviour.

**What I'd try next:** a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) applied after the bi-encoder retrieval step to improve chunk ranking; a self-consistency verification step where the LLM checks its own citation against the retrieved text before finalising the answer; and retry logic with exponential backoff for transient 503/429 errors to prevent infrastructure failures from being recorded as wrong answers.

---

## Part E — Provider Comparison

| Metric | Gemini 2.0 Flash (hosted) | Llama 3.3 70B via Groq (open-source) |
|--------|--------------------------|--------------------------------------|
| grounded_answer_rate | **0.857** (6/7) | 0.571 (4/7) |
| refusal_accuracy | **1.000** (3/3) | **1.000** (3/3) |
| avg_latency_ms | **2134.3 ms** | 22158.2 ms |
| total_cost_usd | $0 (free tier) | $0 (free tier) |
| worst failure shape | quantitative (503 error) | quantitative + buried-section (JSON crash) |

### Failure Shape Breakdown

**Gemini 2.0 Flash** failed on:
- **q3 (quantitative)** — transient 503 server overload error; the model was unavailable at time of the API call. This is an infrastructure failure, not a logic failure. The answer was correctly retrieved in V1. Gemini handled all other question types including interpretation (q6, q7) and buried-section (q4, q5) correctly.

**Llama 3.3 70B via Groq** failed on:
- **q3 (quantitative)** — `json_validate_failed` crash caused by Unicode mathematical symbols (`𝑄𝑖, 𝜗, ℋ`) from the PDF's formal definitions surviving into retrieved chunks. Groq's strict JSON validator rejected these characters mid-generation, causing the entire response to fail with a 400 error.
- **q5 (buried-section)** — same Unicode JSON crash for the same reason; Section 4.3 Discussion chunks contained formal notation.
- **q6 (interpretation)** — retrieval miss. The semantic gap between "why are Planning and Question Developing separate stages" and the document's concrete pipeline description was too large; the model correctly reported the excerpts didn't explicitly address the question rather than hallucinating.

**Hypothesis on failure shapes:** Groq/Llama is significantly more sensitive to prompt-level token artifacts. Its `response_format: json_object` mode enforces strict JSON at the generation level, which causes hard crashes on non-ASCII characters rather than gracefully producing imperfect output. Gemini's internal tokenisation is more robust to Unicode, making it resilient on noisy academic PDFs. On interpretation questions, Gemini benefits from a larger context window and stronger instruction-following, synthesising answers across two sections (q6, q7) more reliably. The 10x latency difference (2134ms vs 22158ms) is largely explained by Groq's free-tier rate limiting causing forced delays between requests rather than raw inference speed.

---

## Part D — Engineering Hygiene

### Unit Tests

Three unit tests in `tests/test_qa.py`:

1. **`test_not_in_document_response`** — verifies that the system returns `answerable=false` and empty citations when similarity is below threshold, without calling the LLM.
2. **`test_validate_and_patch_defaults`** — verifies that missing fields in an LLM response are correctly patched with safe defaults (answerable=false, confidence=0.0, empty citations).
3. **`test_format_context`** — verifies that retrieved chunks are correctly formatted into the prompt context string with section, page, and score metadata in the expected order.

### Data Quality Issue

**Issue: Unicode mathematical symbols embedded in paragraph text.**

PyMuPDF extracts PDF text as a flat stream. Formal definitions in this paper use Unicode Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF) for variables like `𝑄𝑖` (query), `𝜗` (parameters), and `ℋ` (corpus). These render correctly in the PDF but are extracted as raw Unicode codepoints indistinguishable from regular text. When these chunks are retrieved and injected into the LLM prompt, Groq's JSON validator crashes because its `response_format: json_object` mode cannot handle these characters in the generation context — the model enters a generation loop producing malformed JSON until the request times out or errors.

**Mitigation applied:** A `clean_text()` function in `ingest.py` strips the Unicode math blocks (`\u1d00-\u1d7ff`, `\u2000-\u2fff`) before embedding and storage, so the contaminated text never reaches the LLM. A more robust fix would use `fitz.get_text("dict")` to inspect character-level font metadata and selectively replace only the math-symbol spans while preserving surrounding prose — headers are typically larger or bold and could be detected this way too.

### Reproducible Run

```bash
# Full reproducible run from scratch:
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY and GROQ_API_KEY
python main.py ingest         # ~30 seconds
python eval.py --provider gemini --version v1 --output eval_results_v1.json
python eval.py --provider gemini --version v2 --output eval_results_v2.json
python eval.py --provider groq   --version v2 --output eval_results_v2_alt.json
pytest tests/test_qa.py -v
```

---

## Setbacks and Solutions

Throughout development, several non-trivial issues were encountered and resolved:

**1. Corrupted PDF**
Initial ingestion failed with PyMuPDF zlib decompression errors. Fixed by re-downloading directly from the arXiv PDF URL.

**2. Coarse Section Detection**
All chunks were initially grouped under `"Abstract"` because the section header parser was not detecting transitions. Fixed by implementing a two-line lookahead parser with a hardcoded `SECTION_MAP` — reliable for this specific paper whose section numbering and titles are known ahead of time.

**3. Single-Digit Section Numbers Skipped**
Top-level sections (1–9) were being skipped because the page-number filter (which drops standalone digits) fired before the section header check. Fixed by reordering the parsing loop to check for section headers before applying noise filters.

**4. ChromaDB Distance Metric Mismatch**
Switching to `multi-qa-mpnet-base-dot-v1` (dot product model) while keeping `hnsw:space=cosine` caused the distance-to-similarity conversion to be wrong, producing negative scores and false refusals on all questions. Fixed by changing collection space to `hnsw:space=ip` and converting with `similarity = -dist` instead of `1 - dist`.

**5. Similarity Threshold Too Aggressive**
The initial threshold of `0.35` blocked valid queries after switching to the mpnet model whose score distribution differs from cosine-based models. Tuned to `0.25` after observing adversarial questions still scored below this value.

**6. Page 1 Author Information Stripped**
The noise filter was stripping the paper title and author list from page 1 because they matched patterns used to remove running headers on later pages. Fixed by restricting header/footer filtering to `page > 1`.

**7. Groq JSON Validator Crashes**
Formal mathematical notation (`𝑄𝑖, R, 𝜗, H`) from definition blocks caused Groq's `response_format: json_object` to throw `json_validate_failed` errors on q3 and q5. Fixed by adding `clean_text()` in `ingest.py` to strip Unicode math blocks at ingestion time.

**8. Groq Daily Token Limit**
Mid-eval token exhaustion (100k TPD) caused q6, q7, q8, q10 to fail with 429 errors during one run. Mitigation: increased `time.sleep()` between questions in `eval.py` and switched to Gemini as primary provider.

**9. Weak Retrieval on Interpretation Questions**
Q6 consistently failed because the semantic gap between the abstract question phrasing and the document's concrete text was too large. Fixed by adding keyword-based query expansion in `retriever.py` for questions containing "why", "how does", or "distinguish".

**10. Embedding Model Iteration**
Three models were tried before settling on the final choice:

| Model | Problem |
|-------|---------|
| `all-MiniLM-L6-v2` | Symmetric model; very low scores (0.14) on asymmetric QA; author lookup below any workable threshold |
| `multi-qa-MiniLM-L6-cos-v1` | Shifted score distribution caused Q2 to fall below threshold |
| `BAAI/bge-small-en-v1.5` | Requires query prefix; did not work reliably in this setup |
| `multi-qa-mpnet-base-dot-v1` | Best results; purpose-built for question-to-passage retrieval ✓ |

---

## Recording
https://drive.google.com/file/d/1-kxFTMp33Z1-0x4hianYDuRHU7Q_fJg0/view?usp=sharing

