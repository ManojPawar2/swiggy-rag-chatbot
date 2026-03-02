"""
app.py — Flask web server for the Swiggy Annual Report RAG Chatbot.

Serves the HTML chat UI and exposes a /ask endpoint that the frontend
calls via fetch(). The RAG components are initialised once at startup
so every request is fast (no reloading the model per query).

Usage:
    venv\\Scripts\\python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import logging

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load .env before importing config so all env vars are available
load_dotenv()

import config
from rag_pipeline import (
    get_embedding_model,
    get_or_build_vector_store,
    answer_question,
    get_llm,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")

# ─────────────────────────────────────────────────────────────────────────────
# One-time RAG initialisation at startup
# ─────────────────────────────────────────────────────────────────────────────

log.info("Initialising RAG pipeline…")
try:
    embeddings  = get_embedding_model()
    vectorstore = get_or_build_vector_store(config.PDF_PATH, embeddings)
    llm         = get_llm()
    log.info("✅  RAG pipeline ready.")
except FileNotFoundError as e:
    log.error(str(e))
    sys.exit(1)
except Exception as e:
    log.error(f"Initialisation failed: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    Body:    { "question": "What was Swiggy's revenue?" }
    Returns: { "answer": "...", "pages": [12, 34, 56] }
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    log.info(f"Question: {question!r}")

    try:
        answer, pages = answer_question(question, vectorstore, llm)
    except Exception as e:
        log.error(f"RAG error: {e}")
        return jsonify({"error": "Failed to generate answer. Please try again."}), 500

    log.info(f"Answer generated. Pages cited: {pages}")
    return jsonify({"answer": answer, "pages": pages})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print(f"  Swiggy Annual Report AI  —  Web UI")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
