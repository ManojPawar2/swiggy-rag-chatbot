# 📄 Swiggy Annual Report — AI Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot with a modern **Web UI** that answers questions strictly from the Swiggy Annual Report — no hallucinations, no external knowledge.

![Swiggy AI Web UI](https://img.shields.io/badge/UI-Flask%20%2B%20HTML5-FC8019?style=flat-square)
![Embeddings](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en-blue?style=flat-square)

---

## 🏛️ Architecture

### Frontend (Web UI)
- Pure HTML/CSS/JS (no heavy frameworks)
- Animated chat interface with citations and suggested questions
- Connects to the backend via `POST /ask`

### Backend (Flask & RAG Pipeline)
1. **Flask (`app.py`)**: Serves the UI and handles API requests. Loads the ML models *once* at startup for blazing-fast responses.
2. **Document Pipeline**: PyMuPDF + Tesseract OCR (fallback) extracts text from the PDF.
3. **Chunking**: RecursiveCharacterTextSplitter (`chunk_size=800`, `overlap=150`).
4. **Embeddings & Vector Store**: Uses `BAAI/bge-small-en-v1.5` (local) and saves to a persistent `FAISS` index on disk.
5. **LLM**: Grounded prompt strategy with Temperature = 0. Connects to Gemini, Groq, OpenAI, or local Ollama.

---

## ⚙️ Local Setup

### 1. Clone & Environment

```bash
git clone https://github.com/YOUR_USERNAME/swiggy-rag-chatbot.git
cd swiggy-rag-chatbot

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
copy .env.example .env
```
Edit `.env` and set your preferred LLM:
```dotenv
LLM_PROVIDER=gemini        # or groq, openai, ollama
GEMINI_API_KEY=your_key_here
```

### 4. Provide the PDF
Download the Swiggy Annual Report PDF and place it in the project root as `swiggy_annual_report.pdf`.

---

## 🚀 Running the App

### Option 1: Web UI (Recommended)
```bash
python app.py
```
Open **http://localhost:5000** in your browser to use the graphical chat interface.

### Option 2: Command Line (CLI)
```bash
python main.py
```
Drops you into an interactive terminal Q&A session.

---

## 🌐 Deployment (GitHub to Render)

This project is configured for easy free-tier deployment on [Render](https://render.com/).

### Step 1: Push updates to GitHub
Whenever you make changes locally, push them to your repository:
```bash
git add .
git commit -m "Your commit message here"
git push
```

### Step 2: Deploy on Render
1. Go to **[Render Dashboard](https://dashboard.render.com/)** → **New +** → **Web Service**
2. Connect your GitHub repository.
3. Render will auto-detect the `render.yaml` file, or you can manually configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
4. Add your API key in the Render **Environment** settings (e.g., `GEMINI_API_KEY`).
5. Click **Deploy**.

> **Note on Free Tier:** Render's free tier spins down after 15 minutes of inactivity. The first request after a spin-down will take ~30-50 seconds to load the FAISS index and embedding models into memory.

---

## 🛡️ Hallucination Prevention

| Mechanism | How it works |
|---|---|
| **Similarity Threshold** | Irrelevant chunks are discarded before reaching the LLM. |
| **Grounded Prompt** | LLM is explicitly told to answer ONLY from context. |
| **Temperature = 0** | Deterministic output; minimises creative generation. |
| **Page Citations** | Every answer cites specific page numbers from the PDF. |

---

## 🔧 Configuration (`config.py`)

Tune the core engine entirely from one file:

- `CHUNK_SIZE` (default: 800)
- `CHUNK_OVERLAP` (default: 150)
- `TOP_K` (default: 7)
- `SIMILARITY_THRESHOLD` (default: 1.30)
- `LLM_PROVIDER` (gemini, groq, etc.)
