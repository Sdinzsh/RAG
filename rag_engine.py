"""
Vectorless RAG Engine
======================
No embeddings. No vector DB.

Strategy:
  1. Parse PDF → extract pages + detect headings via font-size analysis
  2. Build a Document Tree (hierarchical section index)
  3. For every query:
       a. Ask the LLM to REASON over the tree/TOC → choose relevant sections
       b. Retrieve the full text of those sections (with page numbers)
       c. Ask the LLM to synthesise a final answer from the retrieved content

Inspired by: github.com/Thirumurugan240/vectorless_rag_pageindex
Adapted for: Ollama (local LLMs) + Flask Web UI
"""

from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import fitz          # PyMuPDF
import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

OLLAMA_BASE      = "http://localhost:11434"
DEFAULT_MODEL    = "llama3.2"
TOP_K_SECTIONS   = 4          # max sections fed to final answer
MAX_SECTION_CHARS = 3000      # cap per section text


# ──────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    section_id: str       # e.g. "s_003"
    title: str            # heading text or "Page N"
    level: int            # 1=H1, 2=H2, 3=body/page
    page_start: int       # 1-indexed
    page_end: int         # inclusive
    text: str             # full section text

    def short_repr(self) -> str:
        indent = "  " * (self.level - 1)
        pages = (f"p{self.page_start}" if self.page_start == self.page_end
                 else f"pp{self.page_start}-{self.page_end}")
        title_clean = self.title[:70]
        return f"{indent}[{self.section_id}] {title_clean}  ({pages})"

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "preview": self.text[:200].replace("\n", " ") + "…",
        }


@dataclass
class DocumentTree:
    filename: str
    total_pages: int
    sections: list[Section] = field(default_factory=list)
    _index: dict[str, Section] = field(default_factory=dict, repr=False)

    def build_index(self):
        self._index = {s.section_id: s for s in self.sections}

    def get(self, sid: str) -> Optional[Section]:
        return self._index.get(sid)

    def all_sections(self) -> list[Section]:
        return list(self._index.values())

    def tree_summary(self) -> str:
        """Compact text representation sent to the LLM."""
        lines = [f"Document: {self.filename}  ({self.total_pages} pages)\n"]
        for s in self.sections:
            lines.append(s.short_repr())
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# PDF PARSER — build section tree from headings
# ──────────────────────────────────────────────────────────────

def _heading_threshold(doc: fitz.Document) -> float:
    """Compute font-size threshold above which text is treated as a heading."""
    size_counts: dict[float, int] = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sz = round(span["size"], 1)
                    size_counts[sz] = size_counts.get(sz, 0) + len(span["text"])
    if not size_counts:
        return 14.0
    body = max(size_counts, key=lambda k: size_counts[k])
    return body * 1.18


def build_document_tree(pdf_path: str, filename: str) -> DocumentTree:
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    h_thresh = _heading_threshold(doc)

    # Collect per-page data
    pages_data: list[dict] = []
    for pnum in range(total_pages):
        page = doc[pnum]
        headings: list[str] = []
        text_parts: list[str] = []

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            is_heading_block = False
            spans_text: list[str] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if not t:
                        continue
                    spans_text.append(t)
                    sz    = span["size"]
                    bold  = bool(span.get("flags", 0) & (1 << 4))
                    if sz >= h_thresh or (bold and sz >= h_thresh * 0.9):
                        is_heading_block = True

            joined = " ".join(spans_text)
            if joined:
                text_parts.append(joined)
                if is_heading_block and len(joined) < 120:
                    headings.append(joined)

        pages_data.append({
            "page_num": pnum + 1,
            "text": "\n".join(text_parts),
            "headings": headings,
        })

    doc.close()

    # Build sections by grouping pages under detected headings
    sections: list[Section] = []
    sid_ctr = 0

    def new_sid() -> str:
        nonlocal sid_ctr
        sid_ctr += 1
        return f"s_{sid_ctr:03d}"

    cur_title: Optional[str] = None
    cur_start = 1
    cur_texts: list[str] = []

    for pd in pages_data:
        pnum = pd["page_num"]
        headings = pd["headings"]
        text = pd["text"]

        if headings and cur_texts:
            # flush section
            sections.append(Section(
                section_id=new_sid(),
                title=cur_title or f"Page {cur_start}",
                level=1 if (cur_title and not cur_title.startswith("Page ")) else 3,
                page_start=cur_start,
                page_end=pnum - 1,
                text="\n".join(cur_texts),
            ))
            cur_title = headings[0]
            cur_start = pnum
            cur_texts = [text]
        else:
            if cur_title is None:
                cur_title = headings[0] if headings else f"Page {pnum}"
                cur_start = pnum
            cur_texts.append(text)

    # flush last section
    if cur_texts:
        sections.append(Section(
            section_id=new_sid(),
            title=cur_title or f"Page {cur_start}",
            level=1 if (cur_title and not cur_title.startswith("Page ")) else 3,
            page_start=cur_start,
            page_end=total_pages,
            text="\n".join(cur_texts),
        ))

    # Fallback: no headings found → one section per page
    if all(s.title.startswith("Page ") for s in sections):
        sections = []
        sid_ctr = 0
        for pd in pages_data:
            if pd["text"].strip():
                sections.append(Section(
                    section_id=new_sid(),
                    title=f"Page {pd['page_num']}",
                    level=3,
                    page_start=pd["page_num"],
                    page_end=pd["page_num"],
                    text=pd["text"],
                ))

    tree = DocumentTree(filename=filename, total_pages=total_pages, sections=sections)
    tree.build_index()
    logger.info("Built tree: %d sections from %d pages", len(sections), total_pages)
    return tree


# ──────────────────────────────────────────────────────────────
# OLLAMA HELPERS
# ──────────────────────────────────────────────────────────────

def list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_chat(model: str, messages: list[dict]) -> str:
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_chat_stream(model: str, messages: list[dict]):
    """Yield text tokens from Ollama streaming API."""
    with requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        timeout=180,
        stream=True,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue


# ──────────────────────────────────────────────────────────────
# STEP A — LLM TREE SEARCH
# ──────────────────────────────────────────────────────────────

_TREE_SEARCH_SYS = """\
You are a precise document navigator.
Given a document tree (table of contents with section IDs) and a user question,
identify which section IDs are most relevant to answer the question.

RULES:
- Reply ONLY with a valid JSON array of section_id strings, e.g.: ["s_002","s_007"]
- Pick at most {top_k} section IDs.
- Choose the sections most likely to contain the answer.
- NO explanation, NO prose — JSON array ONLY.
"""

def find_relevant_sections(
    tree: DocumentTree,
    query: str,
    model: str,
    top_k: int = TOP_K_SECTIONS,
) -> list[Section]:
    messages = [
        {"role": "system", "content": _TREE_SEARCH_SYS.format(top_k=top_k)},
        {"role": "user", "content": (
            f"Document Tree:\n{tree.tree_summary()}\n\n"
            f"Question: {query}\n\n"
            f"Return up to {top_k} section IDs as a JSON array."
        )},
    ]

    raw = ollama_chat(model, messages)

    # Robustly extract JSON array
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not match:
        logger.warning("LLM returned no JSON array — using first %d sections", top_k)
        return tree.all_sections()[:top_k]

    try:
        ids = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for: %r", raw)
        return tree.all_sections()[:top_k]

    resolved = [tree.get(sid) for sid in ids if isinstance(sid, str) and tree.get(sid)]
    if not resolved:
        logger.warning("No valid section IDs resolved — using first %d", top_k)
        return tree.all_sections()[:top_k]

    return resolved


# ──────────────────────────────────────────────────────────────
# STEP B — FINAL ANSWER (streaming)
# ──────────────────────────────────────────────────────────────

_ANSWER_SYS = """\
You are a helpful assistant answering questions about a specific document.

Use ONLY the provided document sections to answer.
- Be concise and accurate.
- If the answer is not present in the sections, say so clearly.
- Always cite the page number(s), e.g. (Page 4) or (Pages 12–14).
- Format your response in clean markdown.
"""

def build_context(sections: list[Section]) -> str:
    parts = []
    for s in sections:
        pages = (f"Page {s.page_start}" if s.page_start == s.page_end
                 else f"Pages {s.page_start}–{s.page_end}")
        text = s.text[:MAX_SECTION_CHARS]
        parts.append(
            f"--- [{s.section_id}] {s.title}  ({pages}) ---\n{text}"
        )
    return "\n\n".join(parts)


def answer_query_stream(
    tree: DocumentTree,
    query: str,
    model: str,
    chat_history: list[dict] | None = None,
):
    """
    Full RAG pipeline — yields Server-Sent Event tuples:
      ("sections", list[dict])   — retrieved section metadata
      ("token",    str)          — streamed answer token
      ("done",     "")
    """
    # 1 ── Tree search
    relevant = find_relevant_sections(tree, query, model)
    yield ("sections", [s.to_dict() for s in relevant])

    # 2 ── Build context
    context = build_context(relevant)

    # 3 ── Messages
    messages = [{"role": "system", "content": _ANSWER_SYS}]
    for h in (chat_history or []):
        messages.append(h)
    messages.append({
        "role": "user",
        "content": (
            f"Relevant document sections:\n\n{context}\n\n"
            f"Question: {query}"
        ),
    })

    # 4 ── Stream answer
    for token in ollama_chat_stream(model, messages):
        yield ("token", token)

    yield ("done", "")
