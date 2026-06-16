"""
main.py — CLI entry point.

Commands:
  python main.py ingest [--pdf path/to/paper.pdf]
  python main.py ask "Your question here" [--provider gemini|groq] [--version v1|v2]
"""

import argparse
import json
import sys
from dotenv import load_dotenv

load_dotenv()


def cmd_ingest(args):
    from src.ingest import ingest
    n = ingest(args.pdf)
    print(f"[main] Ingestion complete. {n} chunks indexed.")


def cmd_ask(args):
    from src.providers import get_provider
    from src.qa_system import answer_question

    provider = get_provider(args.provider)
    result = answer_question(
        question=args.question,
        provider=provider,
        prompt_version=args.version,
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest
  python main.py ingest --pdf paper/deep_research_survey.pdf
  python main.py ask "Who is the first author?" --provider gemini --version v2
  python main.py ask "What is the per-query cost?" --provider groq --version v2
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Parse PDF and index into ChromaDB")
    p_ingest.add_argument(
        "--pdf",
        default="paper/deep_research_survey.pdf",
        help="Path to the PDF (default: paper/deep_research_survey.pdf)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # --- ask ---
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "groq"],
        help="LLM provider (default: gemini)",
    )
    p_ask.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Prompt version (default: v2)",
    )
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
