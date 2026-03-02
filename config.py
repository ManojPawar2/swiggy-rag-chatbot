"""
config.py — Central configuration for the Swiggy RAG system.

All tunable parameters live here so you never have to hunt through
multiple files to change chunk size, model names, or retrieval settings.
"""

import os

# ─── PDF Source ───────────────────────────────────────────────────────────────
# Drop the Swiggy Annual Report PDF into the project folder and set this path,
# or pass --pdf <path> at runtime via the CLI.
PDF_PATH = os.getenv("PDF_PATH", "swiggy_annual_report.pdf")

# ─── FAISS Index ──────────────────────────────────────────────────────────────
# Where the vector index is saved/loaded on disk.  Created on first run.
FAISS_INDEX_DIR = "faiss_index"

# ─── OCR (for scanned/image-based PDFs) ───────────────────────────────────────
# Set TESSERACT_CMD to the full path of tesseract.exe on your system.
# The pipeline auto-detects scanned pages and falls back to OCR automatically.
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
OCR_DPI = 300   # Higher = better quality, slower.  300 is the sweet spot.

# ─── Chunking ─────────────────────────────────────────────────────────────────
# chunk_size=800: large enough to capture a full paragraph or table row,
#   small enough that retrieved context stays focused.
# chunk_overlap=150: ~18 % overlap prevents answers from being cut at chunk
#   boundaries (important for financial sentences that span paragraph breaks).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ─── Embedding Model ──────────────────────────────────────────────────────────
# BAAI/bge-small-en-v1.5: top-tier open-source embedding model, runs locally,
# no API key required, ~130 MB download on first use.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ─── Retrieval ────────────────────────────────────────────────────────────────
# TOP_K: number of chunks passed to the LLM as context.
# SIMILARITY_THRESHOLD: cosine distance cutoff (lower = more similar in FAISS L2
#   space). Chunks above this distance are considered "not relevant".
TOP_K = 7
SIMILARITY_THRESHOLD = 1.30  # more lenient for OCR-extracted text (was 0.80)

# ─── LLM Provider ─────────────────────────────────────────────────────────────
# Supported values: "gemini" | "openai" | "ollama" | "groq"
# Reads the API key from environment variables (never hard-code keys).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # default changed to ollama

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash-lite"  # higher free-tier quota than flash

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-3.5-turbo"

# Temperature = 0 → deterministic, minimal creative drift → fewer hallucinations
LLM_TEMPERATURE = 0.0

# ─── Ollama (local, FREE, no API key needed) ───────────────────────────────────
# Ollama is already installed on your system.
# Pull a model first:  ollama pull llama3.2
# Then set LLM_PROVIDER=ollama in .env and restart.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─── Groq (free cloud API, 14,400 req/day, no credit card needed) ────────────────
# Sign up free at https://console.groq.com → create API key → paste below.
# Uses llama-3.3-70b by default — smarter than llama3.2, faster than Gemini.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
