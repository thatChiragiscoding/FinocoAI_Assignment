"""
All prompt templates for the deep-research Q&A system.
V1: Basic grounding rules.
V2: Adds explicit adversarial refusal rules on top of V1.
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — shared by both versions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Your job is to answer questions strictly from the provided document excerpts.
You must NEVER use outside knowledge. If the answer is not in the excerpts, say so.
Always respond with valid JSON only — no markdown, no preamble."""


# ---------------------------------------------------------------------------
# V1 — Basic grounding rules
# ---------------------------------------------------------------------------
QA_PROMPT_V1 = """\
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
{{
  "question": "{question}",
  "answer": "<plain English answer or NOT IN DOCUMENT>",
  "answerable": true,
  "confidence": 0.0,
  "citations": [
    {{
      "section": "<section title>",
      "page": 0,
      "snippet": "<exact verbatim sentence from excerpts>"
    }}
  ],
  "reasoning": "<one sentence>"
}}
"""


# ---------------------------------------------------------------------------
# V2 — Adds explicit adversarial refusal rules
# ---------------------------------------------------------------------------
QA_PROMPT_V2 = """\
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

ADVERSARIAL REFUSAL RULES — if ANY of these apply, immediately set answerable=false:
- The question asks about dollar costs, pricing, or per-query expenses → answerable=false.
- The question compares this paper to systems NOT mentioned in the excerpts (e.g. AlphaFold2, protein folding) → answerable=false.
- The question refers to models not discussed in this paper (e.g. GPT-5, Claude 4, future models) → answerable=false.
- The question asks for a specific fact and no sentence in the excerpts directly states that fact → answerable=false.
- When in doubt, refuse rather than hallucinate.

Respond with ONLY this JSON (no markdown, no extra text):
{{
  "question": "{question}",
  "answer": "<plain English answer or NOT IN DOCUMENT>",
  "answerable": true,
  "confidence": 0.0,
  "citations": [
    {{
      "section": "<section title>",
      "page": 0,
      "snippet": "<exact verbatim sentence from excerpts>"
    }}
  ],
  "reasoning": "<one sentence>"
}}
"""

# Map version strings to prompt templates
PROMPT_VERSIONS = {
    "v1": QA_PROMPT_V1,
    "v2": QA_PROMPT_V2,
}
