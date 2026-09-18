# Local Vectorless RAG

PageIndex-inspired document retrieval with **local Ollama models**. Documents are
parsed into a section tree; the model selects node IDs, reads those sections, and
streams an answer with section and location citations. No embeddings, vector
database, PageIndex account, OpenAI API, or API keys are needed.

```text
Local file → section tree → local LLM selects node IDs → section text → cited answer
```

This adapts the supplied PageIndex tutorial's tree → search → retrieve → answer
workflow. Tree construction uses local document structure and extractive previews;
it does not call the hosted PageIndex service or reproduce its indexing algorithm.

## Run

Python 3.10+ and Ollama are required. Install packages and obtain a local model
before disconnecting from the internet, or transfer them from another machine.

```bash
uv venv
uv pip install -r requirements.txt
# Only if a suitable model is not already installed; requires internet:
ollama pull llama3.1
```

Start Ollama in local-only mode in one terminal:

```bash
OLLAMA_NO_CLOUD=1 ollama serve
```

If Ollama already runs as a service, configure `OLLAMA_NO_CLOUD=1` for that service
and restart it instead of starting a second server. This setting disables Ollama
cloud models and web search; see the [official Ollama FAQ](https://docs.ollama.com/faq#how-do-i-disable-ollama-cloud-features).

Start the app in another terminal:

```bash
.venv/bin/python app.py
# Or choose the initial model explicitly:
RAG_MODEL=qwen2.5:1.5b .venv/bin/python app.py
```

On Windows, use `.venv\Scripts\python.exe app.py` and set environment variables
in your shell. Without uv, use `python -m venv .venv` and the virtual environment's
`python -m pip install -r requirements.txt`.

Open **http://127.0.0.1:5000**, select an installed local model, and upload
`spotify.pdf` or another supported document. No Spotify PDF is bundled. Start
with [examples/music_platform.md](examples/music_platform.md), a fictional
Spotify-like system design, and ask:

- What are the two revenue sources described in this design?
- Which databases does the design use?
- Does the document provide actual monthly active user counts?

The tutorial's displayed tree describes an application design, not an annual
report. Questions about real Spotify financial results or risks require a document
that actually contains that evidence.

## Offline behavior

- PDF, DOCX, Markdown, and text parsing happens on this machine.
- Model requests go only to `127.0.0.1:11434`; environment proxies and HTTP
  redirects are disabled. Cloud-named models and models reporting remote metadata
  are rejected before document content is sent.
- The browser uses a bundled Markdown renderer and system fonts. It has no CDN,
  external font, or remote image dependencies.
- Parsed trees are stored in `cache/`; temporary uploads are deleted after parsing.
- The app binds to loopback with Flask debug mode disabled. It does not download
  packages or models during document processing or chat.

## Retrieval and citations

1. **Build the tree:** PDF font sizes, Word heading styles, Markdown headings,
   and text heading heuristics define sections. Multiple PDF headings on the same
   page and deeply nested Markdown headings remain separate nodes. DOCX tables
   are included in document order. Headingless files use pages or logical blocks.
2. **Cache locally:** a SHA-256-derived content-and-format ID identifies the tree.
   Parser cache versions invalidate old trees; upload again to rebuild them.
3. **Search:** Ollama sees bounded batches of section IDs, ancestor titles, and
   extractive summaries. Every node is considered, including parent text and
   descendants beyond the first 30 roots. Additional rounds rank candidates when
   more than four nodes are selected. Selection uses Ollama's
   [structured JSON output](https://docs.ollama.com/api/chat), with IDs validated
   against the supplied candidates.
4. **Retrieve and answer:** full selected section bodies become answer context.
   The prompt requests section ID, title, and location citations. PDFs use physical
   PDF pages (which can differ from printed page labels). Other formats use
   logical **blocks**, not invented page numbers. The sidebar highlights retrieved
   sections and the response includes source badges. If the model omits references,
   the engine appends a retrieved-source footer from the actual section metadata.
   This footer records retrieval provenance; it does not verify each generated claim.

A valid empty selection produces a no-evidence response without answer generation.
Malformed selection JSON uses a local keyword-overlap fallback. Connection and
model failures appear as errors, rather than being reported as missing evidence.
Standalone questions and topic names start a fresh retrieval. Explicit follow-ups
such as "How does it work?" use recent user questions back to the latest topic;
previous assistant answers are never sent as context for the next answer. The
same question context is used for retrieval and generation. Chat history is held
in memory and resets on restart.

## Use from Python

From the project directory, the tutorial's main steps become:

```python
from rag_engine import build_document_tree, find_relevant_sections, answer_query_stream

# No upload to an external service or indexing poll loop.
tree = build_document_tree("spotify.pdf", "spotify.pdf")
print(tree.tree_summary())
pageindex_tree = tree.nested_nodes()  # node_id, title, summary, text, nodes

query = "What monetization features are described in this document?"
nodes = find_relevant_sections(tree, query, model="qwen2.5:1.5b")
print([node.section_id for node in nodes])

# The complete pipeline performs retrieval itself, then streams the answer.
for kind, data in answer_query_stream(tree, query, model="qwen2.5:1.5b"):
    if kind == "token":
        print(data, end="", flush=True)
```

Choose any installed local model in place of `qwen2.5:1.5b`.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `RAG_MODEL` | `llama3.1` | Initial model; change it in the UI at any time |
| `RAG_NUM_CTX` | `16384` | Ollama context window, in tokens |
| `RAG_MAX_CONTEXT_CHARS` | `24000` | Maximum combined retrieved context, in characters |
| `TOP_K_SECTIONS` in `rag_engine.py` | `4` | Maximum retrieved sections |
| `SEARCH_BATCH_SIZE` in `rag_engine.py` | `16` | Section summaries per search request |

Selected text is **never silently truncated**. If it exceeds the context budget,
ask a narrower question or raise the context budget and model context window
within available memory. Character counts are only an approximate token budget;
non-English text and long histories can use more tokens. Generation and retrieval
quality depend on the installed model; citations do not guarantee correctness.

Scanned PDFs require OCR before upload. PDF headings and table reading order are
heuristic, so inspect the tree for complex layouts. Extractive summaries can miss
important details in long sections. Large trees require multiple model calls.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/models` | Installed Ollama model names |
| POST | `/api/upload` | Multipart `file` and optional `model`; returns session and document IDs |
| GET | `/api/doc/<doc_id>` | Open a new session from a cached tree |
| GET | `/api/doc/<doc_id>/tree` | Export the nested tree, including full text |
| POST | `/api/chat` | JSON `session_id`, `query`, optional `model`; streams SSE |
| GET | `/api/session/<session_id>` | Session metadata |
| POST | `/api/session/<session_id>/clear` | Clear conversation history |

Chat event types are `sections`, `token`, `done`, and `error`. Failed responses are
not saved to conversation history. Uploaded trees survive restarts; sessions do not.

## Tests

No network or running Ollama instance is required for the regression suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
node --test tests/markdown.test.cjs
```

Tests cover PDF/Word/Markdown parsing, caches, deep and large-tree retrieval,
invalid IDs, no-evidence handling, context limits, local model restrictions,
stream errors, API upload/chat/export, and safe offline Markdown rendering.

Inspired by Thiru's PageIndex Vectorless RAG tutorial, supplied with the task.
