import argparse
import os
import shutil
import sys
from dotenv import load_dotenv

load_dotenv()

import config
from rag_pipeline import get_embedding_model, get_or_build_vector_store, answer_question, get_llm


def parse_args():
    parser = argparse.ArgumentParser(description="Swiggy Annual Report RAG Chatbot")
    parser.add_argument("--pdf", type=str, default=None, help="Path to the PDF (overrides config)")
    parser.add_argument("--rebuild", action="store_true", help="Force re-indexing of the PDF")
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_path = args.pdf or config.PDF_PATH

    if args.rebuild and os.path.exists(config.FAISS_INDEX_DIR):
        shutil.rmtree(config.FAISS_INDEX_DIR)
        print(f"[INFO] Removed old index at ./{config.FAISS_INDEX_DIR}/\n")

    print("==========================================================")
    print("      Swiggy Annual Report — RAG Chatbot v1.0")
    print("==========================================================")

    try:
        embeddings = get_embedding_model()
        vectorstore = get_or_build_vector_store(pdf_path, embeddings)
        llm = get_llm()
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        sys.exit(1)

    print("\n✅ System ready. Ask me anything about the Swiggy Annual Report.\n")
    print("-" * 58)

    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[Goodbye!]")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("\n[Goodbye!]")
            break

        try:
            answer, _ = answer_question(query, vectorstore, llm)
            print(f"\n🤖 Bot:\n{answer}")
            print("\n" + "-" * 58)
        except Exception as e:
            print(f"\n[ERROR] Could not generate answer: {e}")


if __name__ == "__main__":
    main()
