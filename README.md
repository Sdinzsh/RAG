# 🌲 Vectorless RAG — Document Intelligence

> **No embeddings. No vector DB. Pure LLM reasoning.**


---

## 🧠 How It Works

```
Traditional Vector RAG
  PDF → chunks → embeddings → cosine similarity → hope it's relevant → answer

Vectorless RAG (this project)
  PDF → document TREE → LLM reasons over tree → picks exact sections → answer
```

### The 3-step pipeline

```
Step 1 — PARSE
  PyMuPDF extracts text page by page.
  Font-size analysis detects headings → builds a hierarchical Document Tree.

Step 2 — TREE SEARCH  (the key insight)
  LLM receives the compact tree (like a Table of Contents with section IDs).
  LLM reasons: "Which sections most likely contain the answer?"
  Returns a JSON array of section_ids.
  → No similarity math. No embeddings. Just reasoning.

Step 3 — ANSWER
  Full text of the chosen sections is retrieved (with page numbers).
  LLM synthesises a cited, markdown-formatted answer.
  Response is streamed token-by-token to the UI.
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- At least one model pulled:

```bash
ollama pull llama3.2        # recommended (fast, good reasoning)
# or
ollama pull mistral
ollama pull phi3
ollama pull gemma2
```

### 1 — Clone / download

```bash
git clone <your-repo>
cd vectorless_rag
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Start Ollama

```bash
ollama serve          # if not already running as a service
```

### 4 — Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🖥️ Web UI Features

| Feature | Detail |
|---------|--------|
| **PDF Upload** | Drag-and-drop or click to browse (up to 50 MB) |
| **Document Tree** | Sidebar shows all sections/pages with page numbers |
| **Section Highlight** | Sections used for each answer are highlighted in the tree |
| **Streaming Chat** | Answers stream token-by-token like ChatGPT |
| **Source Badges** | Each answer shows which sections/pages were used |
| **Multi-turn Chat** | Conversation history maintained per session |
| **Model Selector** | Switch between any Ollama model at the top |
| **Clear History** | Reset conversation context without re-uploading |

---

## 📁 Project Structure

```
vectorless_rag/
├── app.py              ← Flask backend + REST API
├── rag_engine.py       ← Core RAG logic
│   ├── build_document_tree()   ← PDF → Section tree
│   ├── find_relevant_sections() ← LLM tree search
│   └── answer_query_stream()   ← Full RAG pipeline
├── templates/
│   └── index.html      ← Web UI (dark industrial theme)
├── uploads/            ← Uploaded PDFs (auto-created)
└── requirements.txt
```

---

## ⚙️ Configuration

Edit the top of `rag_engine.py`:

```python
OLLAMA_BASE       = "http://localhost:11434"   # Ollama server URL
DEFAULT_MODEL     = "llama3.2"                 # Default model
TOP_K_SECTIONS    = 4                          # Max sections per query
MAX_SECTION_CHARS = 3000                       # Chars per section in context
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `GET`  | `/api/models` | List available Ollama models |
| `POST` | `/api/upload` | Upload PDF + build document tree |
| `POST` | `/api/chat` | Chat query → SSE stream |
| `GET`  | `/api/session/<id>` | Session info |
| `POST` | `/api/session/<id>/clear` | Clear chat history |

---

## 💡 Tips

- **Best models for reasoning**: `llama3.2`, `mistral`, `gemma2`, `phi3`
- **Large PDFs**: The tree search is O(sections) not O(pages) — works well on big docs
- **Structured PDFs** (reports, textbooks): heading detection works best
- **Scanned PDFs**: Won't work — needs OCR pre-processing

