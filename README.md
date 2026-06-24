# 🌲 Vectorless RAG — Document Intelligence

> **No embeddings. No vector DB. No cloud. No API keys. Pure local LLM reasoning.**

This project implements a **Vectorless RAG (Retrieval-Augmented Generation)** pipeline. Instead of dividing documents into arbitrary chunks and matching them via vector embeddings/cosine similarity, it parses documents into a **hierarchical tree structure** and uses LLM reasoning to navigate the tree and find the exact relevant sections.

---

## 🧠 How It Works

```
Traditional Vector RAG:
  PDF ──> chunks ──> embeddings ──> cosine similarity ──> context ──> answer

Vectorless RAG (this project):
  File ──> nested Document Tree ──> LLM reasons over tree ──> picks sections ──> answer
```

### The 4-Step Pipeline

1. **Parse & Structure**: Multi-format parsing (PDF, DOCX, Markdown, TXT) extracts content and identifies logical headings, building a hierarchical nested `DocumentTree` with parent/child links.
2. **Content-Addressed Cache**: Parses are content-addressed (using SHA-1 of file content). Duplicate uploads are resolved instantly from the `./cache/` directory.
3. **Two-Stage Tree Search**:
   - **Stage 1**: The LLM evaluates top-level chapters and selects the most relevant ones.
   - **Stage 2**: The LLM drills down into the sub-sections of the chosen chapters to pick final leaf sections.
   - **Fallback**: If the LLM output is malformed or the API fails, a standard-library BM25-ish keyword search is used.
4. **Context Synthesis & Citation**: The full text of the retrieved sections is compiled as context, and the LLM streams a cited, markdown-formatted answer back to the UI.

---

## 🚀 Quick Start

### Prerequisites
- **Python**: version 3.10+
- **Ollama**: installed and running locally with at least one reasoning model pulled (e.g. `llama3.1`, `llama3.2`, `mistral`, `gemma2`):
  ```bash
  ollama pull llama3.1
  ```

### 1 — Clone the Repository
```bash
git clone https://github.com/Sdinzsh/RAG.git
cd RAG
```

### 2 — Install Dependencies
You can install dependencies using either `pip` or `uv` (recommended for speed).

**Using standard pip:**
```bash
pip install -r requirements.txt
```

**Using uv:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3 — Run the Web App
```bash
python main.py
```
Or if using `uv`:
```bash
uv run main.py
```

Open **`http://localhost:5000`** in your browser.

---

## 🖥️ Web UI Features

- **Multi-format Drag-and-Drop**: Upload PDF, TXT, Markdown, and Word (DOCX) files.
- **Interactive Document Tree Sidebar**: Displays a nested document hierarchy with page numbers. Clicking any section automatically populates the chat input.
- **Dynamic Retrieval Highlights**: The sections selected by the LLM during tree-search are dynamically highlighted in the tree sidebar.
- **Streaming Chat UI**: SSE-based streaming response for real-time token-by-token output.
- **Source Badges**: Displays hoverable badges for sections cited by the LLM. Clicking a badge prompts the LLM to expand on that specific section.
- **Session-based Memory**: Maintains chat history (up to the last 10 turns) to retain context across multi-turn conversations.
- **Model Selector**: Allows switching between any locally available Ollama models dynamically.
- **Instant Cache Indicator**: Displays a `(cached · instant)` badge when reloading previously-uploaded files.

---

## 💾 Document Parsers & Formatting

The engine parses documents page-by-page and structures them hierarchically:

| Format | File Extension | Heading / Hierarchy Detection Strategy |
| :--- | :--- | :--- |
| **PDF** | `.pdf` | Font-size and font-weight analysis via PyMuPDF. Auto-calculates a dynamic heading threshold based on body text. |
| **Word** | `.docx` | Paragraph-style parsing via python-docx. Detects native Word heading styles (Heading 1, 2, 3, etc.). |
| **Markdown** | `.md`, `.markdown` | ATX headings parsing (identifies `#` through `######`). |
| **Plain Text** | `.txt` | Double-newline splits; short, uppercase lines or lines beginning with `Chapter`, `Section`, or `Part` are treated as headings. |

*If a document does not contain recognizable headings, the engine automatically falls back to a page-by-page flat hierarchy.*

---

## 📁 Project Structure

```
RAG/
├── app.py              # Flask server, API endpoints, session store
├── rag_engine.py       # Core RAG pipeline, parsing, caching, tree search, generation
├── requirements.txt    # Python package dependencies
├── pyproject.toml      # Project configuration and metadata
├── templates/
│   └── index.html      # Frontend Web UI (Retro/Dark theme with JetBrains Mono)
├── cache/              # Cached DocumentTree JSON files (created on demand)
└── uploads/            # Temporary storage for uploaded documents (created on demand)
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web-based user interface. |
| `GET` | `/api/models` | Lists all locally installed Ollama models. |
| `POST` | `/api/upload` | Uploads a file, parses it, caches the tree, and opens a chat session. |
| `GET` | `/api/doc/<doc_id>` | Resumes a previously parsed and cached document by its content hash. |
| `POST` | `/api/chat` | Performs the two-stage tree search and streams the answer using Server-Sent Events (SSE). |
| `GET` | `/api/session/<session_id>` | Retrieves active session metadata (filename, page count, model, history length). |
| `POST` | `/api/session/<session_id>/clear` | Wipes the chat history for the specified session. |

---

## ⚙️ Configuration

Key parameters can be configured at the top of `rag_engine.py`:

```python
OLLAMA_BASE       = "http://localhost:11434"   # Ollama server connection URL
DEFAULT_MODEL     = "llama3.1"                # Default model used if unspecified
TOP_K_SECTIONS    = 4                          # Maximum number of sections retrieved as context
MAX_SECTION_CHARS = 3000                       # Character cap per section context limit
CACHE_DIR         = Path(__file__).parent / "cache"  # Directory for caching trees
```

---

## 💡 Performance Tips

1. **Reasoning-focused models** like `llama3.1`, `gemma2`, or `mistral` perform best when navigating the document tree.
2. **Structured documents** with clear hierarchies (e.g. reports, research papers, manuals) yield the best parsing results.
3. **Scanned PDFs** (images only) are not supported. Only machine-readable PDFs (with text layer) are parsed.
4. **Cache reuse**: Files are cached by hash. If you change a document's filename but keep the contents identical, it will load instantly from the cache.

---
