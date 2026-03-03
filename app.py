import os
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

import config
from rag_pipeline import get_embedding_model, get_or_build_vector_store, answer_question, get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

embeddings = None
vectorstore = None
llm = None


def _ensure_rag_loaded():
    global embeddings, vectorstore, llm
    if llm is None:
        log.info("Loading RAG pipeline...")
        embeddings = get_embedding_model()
        vectorstore = get_or_build_vector_store(config.PDF_PATH, embeddings)
        llm = get_llm()
        log.info("RAG pipeline ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        _ensure_rag_loaded()
        answer, pages = answer_question(question, vectorstore, llm)
        return jsonify({"answer": answer, "pages": pages})
    except Exception as e:
        log.error(f"Error handling request: {e}")
        return jsonify({"error": "Failed to generate answer. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
