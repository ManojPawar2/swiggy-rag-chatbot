"""
main.py — CLI entry point for the Swiggy Annual Report RAG chatbot.

Usage:
    python main.py                          # uses PDF_PATH from config.py
    python main.py --pdf path/to/report.pdf # override at runtime
    python main.py --rebuild                # force re-index even if index exists

The program:
  1. Loads (or builds) the FAISS vector index
  2. Drops into an interactive question loop
  3. Types 'exit' or 'quit' to stop
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env before importing config so env vars are available
load_dotenv()

import config
from rag_pipeline import get_embedding_model, get_or_build_vector_store, answer_question, get_llm


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swiggy Annual Report RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --pdf swiggy_annual_report_2024.pdf
  python main.py --rebuild
        """,
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the Swiggy Annual Report PDF (overrides config.PDF_PATH)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-indexing even if a FAISS index already exists on disk",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Startup Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        Swiggy Annual Report — RAG Chatbot  v1.0          ║
║  Answers strictly from the report. No hallucinations.    ║
╚══════════════════════════════════════════════════════════╝
Type your question and press Enter.  Type 'exit' to quit.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve PDF path — CLI arg beats config
    pdf_path = args.pdf or config.PDF_PATH

    # If --rebuild is requested, delete the existing index so it gets recreated
    if args.rebuild:
        import shutil
        index_dir = config.FAISS_INDEX_DIR
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            print(f"[INFO] Removed old index at ./{index_dir}/ — rebuilding…\n")

    print(BANNER)

    # ── Initialise components ─────────────────────────────────────────────────
    try:
        embeddings  = get_embedding_model()
        vectorstore = get_or_build_vector_store(pdf_path, embeddings)
        llm         = get_llm()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Initialisation failed: {e}")
        sys.exit(1)

    print("\n✅  System ready.  Ask me anything about the Swiggy Annual Report.\n")
    print("─" * 60)

    # ── Interactive Q&A loop ──────────────────────────────────────────────────
    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Graceful exit on Ctrl+C or piped input ending
            print("\n\n[Goodbye!]")
            break

        if not query:
            continue  # ignore empty Enter presses

        if query.lower() in {"exit", "quit", "q"}:
            print("\n[Goodbye!]")
            break

        # ── Get answer ────────────────────────────────────────────────────────
        try:
            answer, _pages = answer_question(query, vectorstore, llm)
        except Exception as e:
            print(f"\n[ERROR] Could not generate answer: {e}")
            continue

        print(f"\n🤖 Bot:\n{answer}")
        print("\n" + "─" * 60)


if __name__ == "__main__":
    main()
