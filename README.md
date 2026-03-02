# 📄 Swiggy Annual Report — RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions **strictly and exclusively** from the Swiggy Annual Report — no hallucinations, no external knowledge.

---

## 📥 Report Source

> **Official Swiggy Annual Report (FY 2023–24):**
> [https://www.swiggy.com/corporate/investors](https://www.swiggy.com/corporate/investors)
>
> Direct PDF (BSE filing):
> [https://www.bseindia.com/xml-data/corpfiling/AttachHis/0f1c46c0-b1c1-4b0c-9bb4-5e83a51d8e2a.pdf](https://www.bseindia.com/xml-data/corpfiling/AttachHis/0f1c46c0-b1c1-4b0c-9bb4-5e83a51d8e2a.pdf)

Download the PDF and place it in the project root as `swiggy_annual_report.pdf`.

---

## 🏛️ Architecture Overview

```
PDF File
  │
  ▼
[PyMuPDF Loader]        →  Raw text + page number metadata
  │
  ▼
[Text Cleaner]          →  Remove headers/footers, normalize whitespace
  │
  ▼
[RecursiveCharTextSplitter]  chunk_size=800, overlap=150
  │
  ▼
[HuggingFace Embeddings]  →  BAAI/bge-small-en-v1.5  (local, no API key)
  │
  ▼
[FAISS Vector Store]    →  Persisted to ./faiss_index/ (built once)
  │
  ▼
[Similarity Retrieval]  →  top_k=5, L2 distance threshold=0.80
  │
  ▼
[Grounded Prompt]  +  Retrieved Chunks (with page labels)
  │
  ▼
[LLM: Gemini 1.5 Flash / GPT-3.5-turbo]  temperature=0
  │
  ▼
Answer with page citations  OR  "The information is not available in the provided report."
```

---

## 📁 Project Structure

```
RAG Project Chatbot/
├── main.py              # CLI entry point
├── rag_pipeline.py      # Core pipeline (load → chunk → embed → retrieve → answer)
├── config.py            # All tunable parameters
├── prompts.py           # Grounded prompt template
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── faiss_index/         # Auto-created after first run (FAISS index on disk)
```

---

## ⚙️ Setup

### 1. Clone / Copy Project

```bash
cd "d:\RAG Project Chatbot"
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Edit `.env` and fill in your API key:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
PDF_PATH=swiggy_annual_report.pdf
```

> **Get a free Gemini API key:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 5. Download the PDF

Place the Swiggy Annual Report PDF in the project root as `swiggy_annual_report.pdf`.

---

## 🚀 Running the Chatbot

```bash
# Default — reads PDF_PATH from .env
python main.py

# Override PDF path at runtime
python main.py --pdf path/to/swiggy_annual_report_2024.pdf

# Force re-indexing (if the PDF changes)
python main.py --rebuild
```

**First run** builds the FAISS index (~1–3 minutes depending on PDF size).  
**Subsequent runs** load the saved index in seconds.

---

## 💬 Example Session

```
🔍 You: What is Swiggy's total revenue for FY2024?

🤖 Bot:
According to the report, Swiggy's total revenue from operations for FY2024 was
₹11,247 crore, representing a 34.3% year-on-year growth.
(Source: Page 112, 114)

🔍 You: What is Apple's revenue?

🤖 Bot:
The information is not available in the provided report.
```

---

## 🛡️ Hallucination Prevention

| Mechanism | How it works |
|---|---|
| **Similarity threshold** | Chunks with L2 distance > 0.80 are discarded; if no chunk passes, fallback is returned *without calling the LLM* |
| **Grounded prompt** | LLM is explicitly told: *answer ONLY from context, do NOT use external knowledge* |
| **Temperature = 0** | Deterministic output; minimises creative generation |
| **No conversation history** | Each query is independent; no prior answers pollute the context |
| **Page citations** | Every answer cites page numbers so users can verify |

---

## 🔧 Configuration Cheat Sheet

Open `config.py` to tune these values:

| Parameter | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Characters of overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.80 | Max L2 distance for a chunk to be considered relevant |
| `LLM_TEMPERATURE` | 0.0 | 0 = deterministic; increase for more creative answers |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local HuggingFace embedding model |

---

## 📦 Dependencies

See `requirements.txt`.  Key packages:

- **PyMuPDF** — PDF text extraction
- **LangChain** — pipeline orchestration
- **sentence-transformers** — local embeddings
- **faiss-cpu** — vector similarity search
- **langchain-google-genai** — Gemini LLM (if using Gemini)
- **langchain-openai** — OpenAI LLM (if using OpenAI)
- **python-dotenv** — `.env` file loading

---

## ⚠️ Known Limitations

- Only works with the single PDF specified at startup.
- Image-heavy pages (charts, infographics) yield little text — the system gracefully skips near-empty pages.
- Table extraction quality depends on PDF encoding; some tables may be linearised.
