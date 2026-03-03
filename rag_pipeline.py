import os
import re
import sys
import logging
import urllib.request
import json
import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config
from prompts import GROUNDED_PROMPT_TEMPLATE, FALLBACK_RESPONSE


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _setup_tesseract():
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        log.warning(f"Tesseract not available: {e}. OCR disabled.")
        return False


def _ocr_page(page):
    import pytesseract
    from PIL import Image
    import io

    mat = fitz.Matrix(config.OCR_DPI / 72, config.OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang="eng", config="--psm 3")


def clean_text(raw_text):
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"(?m)^.*?\|\s*\d+\s*$", "", text)
    text = re.sub(r"[-_=]{4,}", "", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r" {2,}", " ", text).strip()


def load_and_clean_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log.info(f"Loading PDF: {pdf_path}")
    ocr_available = _setup_tesseract()
    docs = []
    ocr_pages = 0

    with fitz.open(pdf_path) as pdf:
        log.info(f"  -> {len(pdf)} pages detected")

        for page_num, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)

            if len(cleaned) < 80 and ocr_available:
                log.info(f"  -> Page {page_num}: scanned, running OCR...")
                cleaned = clean_text(_ocr_page(page))
                if len(cleaned) >= 50:
                    ocr_pages += 1
                else:
                    continue

            if len(cleaned) >= 50:
                docs.append(Document(
                    page_content=cleaned,
                    metadata={"source": os.path.basename(pdf_path), "page_number": page_num}
                ))

    log.info(f"  -> {len(docs)} pages loaded ({ocr_pages} via OCR)")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(f"  -> {len(chunks)} chunks created")
    return chunks


def get_embedding_model():
    log.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store(embeddings):
    log.info(f"Loading FAISS index from ./{config.FAISS_INDEX_DIR}/")
    return FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_vector_store(chunks, embeddings):
    log.info("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.FAISS_INDEX_DIR)
    log.info(f"  -> Index saved to ./{config.FAISS_INDEX_DIR}/")
    return vectorstore


def get_or_build_vector_store(pdf_path, embeddings):
    index_path = os.path.join(config.FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(index_path):
        return load_vector_store(embeddings)

    docs = load_and_clean_pdf(pdf_path)
    chunks = split_documents(docs)
    return build_vector_store(chunks, embeddings)


def retrieve_chunks(query, vectorstore):
    results = vectorstore.similarity_search_with_score(query, k=config.TOP_K)
    relevant = [doc for doc, score in results if score <= config.SIMILARITY_THRESHOLD]
    
    if relevant:
        pages = [str(d.metadata.get("page_number", "?")) for d in relevant]
        log.info(f"  -> {len(relevant)} chunks retrieved (pages: {', '.join(pages)})")
    else:
        log.info("  -> No chunks passed similarity threshold")
        
    return relevant


def format_context(chunks):
    parts = [f"[Page {doc.metadata.get('page_number', '?')}]\n{doc.page_content}" for doc in chunks]
    return "\n\n---\n\n".join(parts)


def get_llm():
    provider = config.LLM_PROVIDER.lower()

    class PromptMsg:
        def __init__(self, content):
            self.content = content

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            sys.exit("ERROR: GEMINI_API_KEY is not set.")

        from google import genai
        from google.genai import types
        import time

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        log.info(f"Using LLM: Google {config.GEMINI_MODEL}")

        class GeminiLLM:
            def invoke(self, prompt):
                for attempt in range(1, 4):
                    try:
                        res = client.models.generate_content(
                            model=config.GEMINI_MODEL,
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=config.LLM_TEMPERATURE)
                        )
                        return PromptMsg(res.text)
                    except Exception as e:
                        if attempt < 3 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                            log.warning(f"Rate limited, retrying in {20 * attempt}s...")
                            time.sleep(20 * attempt)
                            continue
                        raise
                raise RuntimeError("Gemini API rate limit exhausted.")
        return GeminiLLM()

    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            sys.exit("ERROR: OPENAI_API_KEY is not set.")
            
        from langchain_openai import ChatOpenAI
        log.info(f"Using LLM: OpenAI {config.OPENAI_MODEL}")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
        )

    elif provider == "ollama":
        base_url = config.OLLAMA_BASE_URL.rstrip("/")
        log.info(f"Using LLM: Ollama ({config.OLLAMA_MODEL}) at {base_url}")

        class OllamaLLM:
            def invoke(self, prompt):
                payload = json.dumps({
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": config.LLM_TEMPERATURE},
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{base_url}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        return PromptMsg(data.get("response", "").strip())
                except Exception as e:
                    raise RuntimeError(f"Ollama request failed: {e}")
        return OllamaLLM()

    elif provider == "groq":
        if not config.GROQ_API_KEY:
            sys.exit("ERROR: GROQ_API_KEY is not set.")

        from openai import OpenAI
        client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
        log.info(f"Using LLM: Groq ({config.GROQ_MODEL})")

        class GroqLLM:
            def invoke(self, prompt):
                res = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=1024,
                )
                return PromptMsg(res.choices[0].message.content.strip())
        return GroqLLM()

    else:
        sys.exit(f"ERROR: Unknown LLM_PROVIDER '{provider}'.")


def answer_question(query, vectorstore, llm):
    chunks = retrieve_chunks(query, vectorstore)
    if not chunks:
        return FALLBACK_RESPONSE, []

    context = format_context(chunks)
    prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context, question=query)

    log.info("Calling LLM...")
    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)
    pages = sorted({doc.metadata["page_number"] for doc in chunks if doc.metadata.get("page_number")})

    return answer.strip(), pages
