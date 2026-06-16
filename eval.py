"""
eval.py — Run all 10 eval questions and compute metrics.

Usage:
  python eval.py --provider gemini --version v1 --output eval_results_v1.json
  python eval.py --provider gemini --version v2 --output eval_results_v2.json
  python eval.py --provider groq   --version v2 --output eval_results_v2_alt.json

Metrics printed:
  1. grounded_answer_rate   — fraction of answerable Qs (q1-q7) where citation
                              section actually contains the answer (self-graded).
  2. refusal_accuracy       — fraction of unanswerable Qs (q8-q10) correctly
                              returned as answerable=false.
  3. avg_confidence_correct — mean confidence on questions graded correct.
  4. avg_confidence_wrong   — mean confidence on questions graded wrong.
  5. avg_latency_ms         — mean latency per question.
"""

import argparse
import json
import time
from dotenv import load_dotenv

load_dotenv()

# q1-q7 are answerable; q8-q10 are adversarial
ANSWERABLE_IDS = {"q1", "q2", "q3", "q4", "q5", "q6", "q7"}
UNANSWERABLE_IDS = {"q8", "q9", "q10"}


# ---------------------------------------------------------------------------
# Manual grading helper
# ---------------------------------------------------------------------------
def auto_grade(qid: str, result: dict) -> bool:
    """
    Lightweight automatic grading for answerable questions.
    Returns True if the result looks correct:
      - answerable=True
      - at least one non-empty citation
      - answer is not "NOT IN DOCUMENT"

    This is a proxy — the README reminds the user to manually verify.
    """
    if not result.get("answerable", False):
        return False
    if not result.get("citations"):
        return False
    answer = result.get("answer", "").strip().upper()
    if answer == "NOT IN DOCUMENT" or answer == "":
        return False
    return True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(questions: list, results: list) -> dict:
    q_map = {q["id"]: q for q in questions}

    grounded_correct = 0
    grounded_total = 0
    refusal_correct = 0
    refusal_total = 0

    confidence_correct = []
    confidence_wrong = []
    latencies = []

    for result in results:
        qid = result.get("id", "")
        latencies.append(result.get("latency_ms", 0))
        conf = float(result.get("confidence", 0.0))

        if qid in ANSWERABLE_IDS:
            grounded_total += 1
            correct = auto_grade(qid, result)
            result["_auto_graded_correct"] = correct
            if correct:
                grounded_correct += 1
                confidence_correct.append(conf)
            else:
                confidence_wrong.append(conf)

        elif qid in UNANSWERABLE_IDS:
            refusal_total += 1
            refused = not result.get("answerable", True)
            result["_refused_correctly"] = refused
            if refused:
                refusal_correct += 1
                confidence_correct.append(conf)  # correct = refusing when should refuse
            else:
                confidence_wrong.append(conf)

    grounded_answer_rate = grounded_correct / grounded_total if grounded_total else 0.0
    refusal_accuracy = refusal_correct / refusal_total if refusal_total else 0.0
    avg_confidence_correct = sum(confidence_correct) / len(confidence_correct) if confidence_correct else 0.0
    avg_confidence_wrong = sum(confidence_wrong) / len(confidence_wrong) if confidence_wrong else 0.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "grounded_answer_rate": round(grounded_answer_rate, 3),
        "refusal_accuracy": round(refusal_accuracy, 3),
        "avg_confidence_correct": round(avg_confidence_correct, 3),
        "avg_confidence_wrong": round(avg_confidence_wrong, 3),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "grounded_correct": grounded_correct,
        "grounded_total": grounded_total,
        "refusal_correct": refusal_correct,
        "refusal_total": refusal_total,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
def run_eval(provider_name: str, version: str, output_path: str, questions_path: str):
    from src.providers import get_provider
    from src.qa_system import answer_question

    # Load questions
    with open(questions_path) as f:
        questions = json.load(f)

    provider = get_provider(provider_name)
    results = []

    print(f"\n{'='*60}")
    print(f"  Provider : {provider_name.upper()}")
    print(f"  Version  : {version}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        print(f"[{i:02d}/{len(questions)}] {qid}: {question[:70]}...")

        result = answer_question(
            question=question,
            provider=provider,
            prompt_version=version,
        )
        result["id"] = qid
        result["shape"] = q.get("shape", "")

        answerable_str = "YES answerable" if result["answerable"] else "NO refused"
        print(f"         -> {answerable_str} | conf={result['confidence']:.2f} | {result['latency_ms']}ms\n")

        results.append(result)

        # Sleep between questions to respect rate limits
        if i < len(questions):
            time.sleep(1)

    # Compute metrics
    metrics = compute_metrics(questions, results)

    # Build output object
    output = {
        "metadata": {
            "provider": provider_name,
            "version": version,
            "total_questions": len(questions),
        },
        "metrics": metrics,
        "results": results,
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SAVED -> {output_path}")
    print(f"{'='*60}")
    print(f"  grounded_answer_rate   : {metrics['grounded_answer_rate']:.3f}  ({metrics['grounded_correct']}/{metrics['grounded_total']})")
    print(f"  refusal_accuracy       : {metrics['refusal_accuracy']:.3f}  ({metrics['refusal_correct']}/{metrics['refusal_total']})")
    print(f"  avg_confidence_correct : {metrics['avg_confidence_correct']:.3f}")
    print(f"  avg_confidence_wrong   : {metrics['avg_confidence_wrong']:.3f}")
    print(f"  avg_latency_ms         : {metrics['avg_latency_ms']:.1f} ms")
    print(f"{'='*60}\n")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the Q&A system on eval_questions.json")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--output", default="eval_results_v2.json")
    parser.add_argument("--questions", default="eval_questions.json")
    args = parser.parse_args()

    run_eval(
        provider_name=args.provider,
        version=args.version,
        output_path=args.output,
        questions_path=args.questions,
    )


if __name__ == "__main__":
    main()
