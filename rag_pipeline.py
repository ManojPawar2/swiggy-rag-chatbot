"""
rag_pipeline.py — Core RAG pipeline for the Swiggy Annual Report chatbot.

Functions are intentionally kept flat and readable.  Each function does one
thing, takes explicit arguments, and returns a clear value — no magic globals.

Pipeline flow:
  load_and_clean_pdf  →  build_vector_store  →  retrieve_chunks  →  answer_question
"""

import os
import re
import sys
import logging

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config
from prompts import GROUNDED_PROMPT_TEMPLATE, FALLBACK_RESPONSE

# Set up a simple logger — production systems need logs, not bare print()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Loading & Cleaning (supports both text-based and scanned PDFs)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_tesseract() -> bool:
    """
    Point pytesseract at the Tesseract binary from config.
    Returns True if Tesseract is available, False otherwise.
    """
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        # Quick sanity check — will raise if binary not found
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        log.warning(f"Tesseract not available ({e}). OCR disabled.")
        return False


def _ocr_page(page: fitz.Page) -> str:
    """
    Render a PDF page to a high-res image and run Tesseract OCR on it.
    Used as fallback for scanned / image-only pages.
    """
    import pytesseract
    from PIL import Image
    import io

    # Render page at OCR_DPI — higher DPI = better accuracy, more RAM
    mat = fitz.Matrix(config.OCR_DPI / 72, config.OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)  # grey is faster
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    # Tesseract with English language model
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 3")
    return text


def clean_text(raw: str) -> str:
    """
    Light-touch text cleaning after extracting from PDF or OCR.
    """
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"(?m)^.*?\|\s*\d+\s*$", "", text)
    text = re.sub(r"[-_=]{4,}", "", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_and_clean_pdf(pdf_path: str) -> list[Document]:
    """
    Load each page of the PDF and extract text.

    For text-based PDFs  → direct extraction via PyMuPDF (fast).
    For scanned PDFs     → renders each page to an image and runs Tesseract OCR.
    Mixed PDFs           → uses whichever method works per-page automatically.

    Returns a list of Document objects with page-number metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: '{pdf_path}'\n"
            f"Place the Swiggy Annual Report PDF in the project folder "
            f"and update PDF_PATH in config.py (or pass --pdf <path>)."
        )

    log.info(f"Loading PDF: {pdf_path}")
    ocr_available = _setup_tesseract()
    docs = []
    ocr_pages = 0

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)
        log.info(f"  → {total_pages} pages detected")

        for page_num, page in enumerate(pdf, start=1):

            # Step 1: try fast text extraction first
            raw_text = page.get_text("text")
            cleaned  = clean_text(raw_text)

            # Step 2: if page looks scanned (barely any text), fall back to OCR
            if len(cleaned) < 80:
                if ocr_available:
                    log.info(f"  → Page {page_num}: scanned — running OCR…")
                    raw_ocr = _ocr_page(page)
                    cleaned = clean_text(raw_ocr)
                    if len(cleaned) >= 50:
                        ocr_pages += 1
                    else:
                        continue   # OCR got nothing useful either — skip page
                else:
                    continue       # No OCR available, skip blank/image page

            docs.append(Document(
                page_content=cleaned,
                metadata={
                    "source":      os.path.basename(pdf_path),
                    "page_number": page_num,
                }
            ))

    log.info(
        f"  → {len(docs)} pages loaded "
        f"({ocr_pages} via OCR, {len(docs) - ocr_pages} via text extraction)"
    )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunking
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split page-level documents into smaller, overlapping chunks.

    Why RecursiveCharacterTextSplitter?
      - It tries to split at natural boundaries first (paragraphs → sentences
        → words) before resorting to character-level splits.
      - This keeps financial sentences and table rows intact wherever possible.

    chunk_size=800, overlap=150 chosen after empirical testing on annual
    report content — captures full paragraphs without drowning the LLM with
    too much context per chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # prefer paragraph breaks
    )

    chunks = splitter.split_documents(docs)

    # Re-attach the page_number metadata so it survives the split
    # (LangChain propagates metadata automatically, but we log it for sanity)
    log.info(f"  → {len(chunks)} chunks created from {len(docs)} pages")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embeddings & Vector Store
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model once and return it.

    BAAI/bge-small-en-v1.5 is our choice because:
      - It scores near the top of the MTEB leaderboard for retrieval tasks
      - It's only ~130 MB — fast to download and runs on CPU without issue
      - No API key required — fully local
    """
    log.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
    )


def build_vector_store(chunks: list[Document],
                       embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Create a FAISS vector store from document chunks and persist it locally.

    Why FAISS?
      - Zero infrastructure: no server, no API key, no Docker
      - Fast enough for a single annual report (< 10k chunks)
      - Deterministic — same query always returns same results
      - Persisted to disk so re-indexing is only needed once
    """
    log.info("Building FAISS vector store…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.FAISS_INDEX_DIR)
    log.info(f"  → Index saved to ./{config.FAISS_INDEX_DIR}/")

    return vectorstore


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    Returns the vectorstore ready for querying.
    """
    log.info(f"Loading FAISS index from ./{config.FAISS_INDEX_DIR}/")
    vectorstore = FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe — we wrote this index ourselves
    )
    return vectorstore


def get_or_build_vector_store(pdf_path: str,
                               embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Convenience wrapper: build a fresh index if one doesn't exist yet,
    otherwise load the saved one.  This is the main entry point for the
    vector store.
    """
    index_exists = os.path.exists(
        os.path.join(config.FAISS_INDEX_DIR, "index.faiss")
    )

    if index_exists:
        return load_vector_store(embeddings)

    # First run — need to load the PDF and index it
    docs   = load_and_clean_pdf(pdf_path)
    chunks = split_documents(docs)
    return build_vector_store(chunks, embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, vectorstore: FAISS) -> list[Document]:
    """
    Retrieve the top-K most semantically similar chunks for a query.

    Uses FAISS similarity_search_with_score which returns (document, L2_distance)
    pairs.  We filter by SIMILARITY_THRESHOLD to avoid passing totally irrelevant
    chunks to the LLM — this is the first line of hallucination defence.

    Lower L2 distance = more similar (threshold is a maximum allowed distance).
    """
    results = vectorstore.similarity_search_with_score(
        query, k=config.TOP_K
    )

    # Filter out chunks whose distance exceeds our threshold
    relevant = [
        doc for doc, score in results
        if score <= config.SIMILARITY_THRESHOLD
    ]

    if relevant:
        pages = [str(d.metadata.get("page_number", "?")) for d in relevant]
        log.info(f"  → {len(relevant)} relevant chunks retrieved (pages: {', '.join(pages)})")
    else:
        log.info("  → No chunks passed the similarity threshold")

    return relevant


def format_context(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.
    Each chunk is labelled with its page number so the LLM can cite it.
    """
    parts = []
    for i, doc in enumerate(chunks, start=1):
        page = doc.metadata.get("page_number", "?")
        parts.append(f"[Chunk {i} — Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM Setup
# ─────────────────────────────────────────────────────────────────────────────

def get_llm():
    """
    Return a configured LLM instance based on LLM_PROVIDER in config.

    Supports:
      - "gemini" → Google Gemini via google-genai SDK (direct, no wrapper issues)
      - "openai" → OpenAI GPT-3.5-turbo via langchain-openai

    Temperature is forced to 0 for deterministic, grounded responses.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            sys.exit(
                "ERROR: GEMINI_API_KEY is not set.  "
                "Add it to your .env file or export it as an environment variable."
            )

        # Use google-genai SDK directly — more reliable than the langchain wrapper
        from google import genai as google_genai
        from google.genai import types as genai_types

        client = google_genai.Client(api_key=config.GEMINI_API_KEY)
        model_name = config.GEMINI_MODEL
        temperature = config.LLM_TEMPERATURE

        log.info(f"Using LLM: Google {model_name}")

        # Tiny wrapper so the rest of the code can call llm.invoke(prompt)
        # and get back an object with a .content attribute — same interface
        # as LangChain chat models.
        class _GeminiLLM:
            def invoke(self, prompt: str):
                import time

                max_retries = 3
                wait_secs   = 35  # default wait if API doesn't tell us

                for attempt in range(1, max_retries + 1):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=genai_types.GenerateContentConfig(
                                temperature=temperature,
                            ),
                        )
                        class _Msg:
                            content = response.text
                        return _Msg()

                    except Exception as e:
                        err = str(e)
                        # Check if it's a rate-limit error worth retrying
                        if "429" in err or "RESOURCE_EXHAUSTED" in err:
                            if attempt < max_retries:
                                log.warning(
                                    f"Rate limit hit (attempt {attempt}/{max_retries}). "
                                    f"Waiting {wait_secs}s before retry…"
                                )
                                time.sleep(wait_secs)
                                wait_secs *= 2  # exponential backoff
                                continue
                        raise  # re-raise anything that's not a rate-limit

                raise RuntimeError("Gemini API rate limit: all retries exhausted.")

        return _GeminiLLM()

    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            sys.exit(
                "ERROR: OPENAI_API_KEY is not set.  "
                "Add it to your .env file or export it as an environment variable."
            )
        from langchain_openai import ChatOpenAI
        log.info(f"Using LLM: OpenAI {config.OPENAI_MODEL}")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
        )

    elif provider == "ollama":
        # Ollama runs locally — no API key, no rate limits, completely free.
        # Make sure Ollama is running and the model is pulled:
        #   ollama pull llama3.2
        import urllib.request
        import json as _json

        base_url  = config.OLLAMA_BASE_URL.rstrip("/")
        model_name = config.OLLAMA_MODEL

        log.info(f"Using LLM: Ollama ({model_name}) at {base_url}")

        class _OllamaLLM:
            def invoke(self, prompt: str):
                payload = _json.dumps({
                    "model":  model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": config.LLM_TEMPERATURE},
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{base_url}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = _json.loads(resp.read().decode("utf-8"))
                except Exception as e:
                    raise RuntimeError(
                        f"Ollama request failed: {e}\n"
                        f"Make sure Ollama is running: run 'ollama serve' in a terminal."
                    )

                class _Msg:
                    content = data.get("response", "").strip()

                return _Msg()

        return _OllamaLLM()

    elif provider == "groq":
        # Groq cloud API — free tier, 14,400 req/day, OpenAI-compatible.
        # Uses the openai package with Groq's base URL — avoids Cloudflare blocks.
        if not config.GROQ_API_KEY:
            sys.exit(
                "ERROR: GROQ_API_KEY is not set. "
                "Sign up free at https://console.groq.com and add it to your .env file."
            )

        from openai import OpenAI as _OpenAI

        _groq_client = _OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
        model_name = config.GROQ_MODEL

        log.info(f"Using LLM: Groq ({model_name})")

        class _GroqLLM:
            def invoke(self, prompt: str):
                resp = _groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=1024,
                )
                class _Msg:
                    content = resp.choices[0].message.content.strip()
                return _Msg()

        return _GroqLLM()

    else:
        sys.exit(f"ERROR: Unknown LLM_PROVIDER '{provider}'. Use 'gemini', 'openai', 'ollama', or 'groq'.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(query: str, vectorstore: FAISS, llm) -> tuple:
    """
    Full RAG chain: retrieve relevant chunks → build grounded prompt → call LLM.

    Hallucination prevention checkpoints:
      1. If no chunks pass the similarity threshold → return FALLBACK_RESPONSE
         immediately, without ever calling the LLM.
      2. The prompt template forces the LLM to answer ONLY from context.
      3. Temperature = 0 limits creative generation.

    Returns a tuple of (answer_string, cited_page_numbers_list).
    """
    # Step 1: Retrieve
    relevant_chunks = retrieve_chunks(query, vectorstore)

    # Fallback: no relevant context found
    if not relevant_chunks:
        return FALLBACK_RESPONSE, []

    # Step 2: Format context block with page labels
    context = format_context(relevant_chunks)

    # Step 3: Build the grounded prompt
    prompt = GROUNDED_PROMPT_TEMPLATE.format(
        context=context,
        question=query,
    )

    # Step 4: Call the LLM
    log.info("Calling LLM…")
    response = llm.invoke(prompt)

    # LangChain chat models return an AIMessage; extract the text
    answer = response.content if hasattr(response, "content") else str(response)

    # Collect unique page numbers from retrieved chunks, sorted
    pages = sorted({
        doc.metadata["page_number"]
        for doc in relevant_chunks
        if doc.metadata.get("page_number")
    })

    return answer.strip(), pages
