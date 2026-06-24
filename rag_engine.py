"""
Vectorless RAG Engine
======================
No embeddings. No vector DB. No cloud. No API keys.

Strategy:
  1. Parse a local file (PDF / TXT / MD / DOCX) → extract pages + detect headings
  2. Build a NESTED Document Tree (hierarchical section index with parent/child links)
  3. For every query:
       a. STAGE 1 — Ask the LLM to reason over top-level chapters → pick chapters
       b. STAGE 2 — For each chosen chapter, drill into its children → pick leaf sections
       c. Retrieve the full text of those sections (with page numbers)
       d. Ask the LLM to synthesise a cited final answer from the retrieved content

Parsed trees are cached to disk keyed by a content hash (doc_id), so re-uploading
the same file loads instantly without re-parsing.

Inspired by: github.com/Thirumurugan240/vectorless_rag_pageindex
Adapted for: Ollama (local LLMs) + Flask Web UI — 100% offline.
"""

from __future__ import annotations

import re
import json
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

OLLAMA_BASE       = "http://localhost:11434"
DEFAULT_MODEL     = "llama3.1"
TOP_K_SECTIONS    = 4          # max sections fed to final answer
MAX_SECTION_CHARS = 3000       # cap per section text
CACHE_DIR         = Path(__file__).parent / "cache"

SUPPORTED_EXT = (".pdf", ".txt", ".md", ".markdown", ".docx")

# ──────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    section_id: str                       # e.g. "s_003"
    title: str                            # heading text or "Page N"
    level: int                            # 1=H1, 2=H2, 3=H3/body
    page_start: int                       # 1-indexed
    page_end: int                         # inclusive
    text: str                             # full section text
    parent_id: Optional[str] = None       # parent section_id (None for roots)
    children: list[str] = field(default_factory=list)

    def short_repr(self) -> str:
        indent = "  " * (self.level - 1)
        pages = (f"p{self.page_start}" if self.page_start == self.page_end
                 else f"pp{self.page_start}-{self.page_end}")
        title_clean = self.title[:70]
        return f"{indent}[{self.section_id}] {title_clean}  ({pages})"

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title":      self.title,
            "level":      self.level,
            "page_start": self.page_start,
            "page_end":   self.page_end,
            "preview":    self.text[:200].replace("\n", " ") + "\u2026",
            "children":   list(self.children),
        }

    def to_summary_dict(self) -> dict:
        """Compact form for the LLM table-of-contents prompt."""
        pages = (f"p{self.page_start}" if self.page_start == self.page_end
                 else f"pp{self.page_start}-{self.page_end}")
        return {
            "id":      self.section_id,
            "title":   self.title[:80],
            "pages":   pages,
            "preview": self.text[:160].replace("\n", " ").strip(),
        }


@dataclass
class DocumentTree:
    filename: str
    total_pages: int
    sections: list[Section] = field(default_factory=list)
    doc_id: str = ""
    source_format: str = ""
    from_cache: bool = False
    roots: list[str] = field(default_factory=list)
    _index: dict[str, Section] = field(default_factory=dict, repr=False)

    def build_index(self):
        self._index = {s.section_id: s for s in self.sections}
        if not self.roots:
            self.roots = [s.section_id for s in self.sections if s.parent_id is None]

    def get(self, sid: str) -> Optional[Section]:
        return self._index.get(sid)

    def children_of(self, sid: str) -> list[Section]:
        sec = self._index.get(sid)
        if not sec:
            return []
        return [self._index[c] for c in sec.children if c in self._index]

    def roots_sections(self) -> list[Section]:
        return [self._index[s] for s in self.roots if s in self._index]

    def all_sections(self) -> list[Section]:
        return list(self._index.values())

    def walk(self) -> Iterator[Section]:
        """Depth-first walk of the tree (roots first, then their children)."""
        seen = set()
        def _walk(sid: str):
            if sid in seen or sid not in self._index:
                return
            seen.add(sid)
            yield self._index[sid]
            for c in self._index[sid].children:
                yield from _walk(c)
        for r in self.roots:
            yield from _walk(r)
        # any orphans (e.g. flat fallback) emitted in declaration order
        for s in self.sections:
            if s.section_id not in seen:
                yield s

    def tree_summary(self) -> str:
        """Compact text representation sent to the LLM."""
        lines = [f"Document: {self.filename}  ({self.total_pages} pages)\n"]
        for s in self.walk():
            lines.append(s.short_repr())
        return "\n".join(lines)

    # ── cache (de)serialisation ───────────────────────────────
    def to_cache_dict(self) -> dict:
        return {
            "filename":      self.filename,
            "total_pages":   self.total_pages,
            "doc_id":        self.doc_id,
            "source_format": self.source_format,
            "sections":      [self._sec_to_raw(s) for s in self.sections],
        }

    @staticmethod
    def _sec_to_raw(s: Section) -> dict:
        return {
            "section_id": s.section_id,
            "title":      s.title,
            "level":      s.level,
            "page_start": s.page_start,
            "page_end":   s.page_end,
            "text":       s.text,
            "parent_id":  s.parent_id,
            "children":   list(s.children),
        }

    @classmethod
    def from_cache_dict(cls, d: dict) -> "DocumentTree":
        sections = [Section(
            section_id=r["section_id"],
            title=r["title"],
            level=r["level"],
            page_start=r["page_start"],
            page_end=r["page_end"],
            text=r["text"],
            parent_id=r.get("parent_id"),
            children=list(r.get("children", [])),
        ) for r in d["sections"]]
        tree = cls(
            filename=d["filename"],
            total_pages=d["total_pages"],
            sections=sections,
            doc_id=d.get("doc_id", ""),
            source_format=d.get("source_format", ""),
        )
        tree.build_index()
        return tree


# ──────────────────────────────────────────────────────────────
# FILE PARSER LAYER  (dispatch by extension)
# ──────────────────────────────────────────────────────────────

def parse_file(path: str) -> tuple[list[dict], str]:
    """
    Parse a local file into per-page dicts:
        { "page_num": int, "text": str, "headings": list[str] }
    Returns (pages, source_format). Raises ValueError on unsupported types.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path), "pdf"
    if ext in (".txt",):
        with open(path, encoding="utf-8", errors="replace") as f:
            return _parse_text(f.read()), "txt"
    if ext in (".md", ".markdown"):
        with open(path, encoding="utf-8", errors="replace") as f:
            return _parse_markdown(f.read()), "md"
    if ext == ".docx":
        return _parse_docx(path), "docx"
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXT)}"
    )


def _parse_pdf(pdf_path: str) -> list[dict]:
    """PyMuPDF + font-size heading detection (unchanged behaviour)."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    h_thresh = _heading_threshold(doc)

    pages_data: list[dict] = []
    for pnum in range(doc.page_count):
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
                    sz   = span["size"]
                    bold = bool(span.get("flags", 0) & (1 << 4))
                    if sz >= h_thresh or (bold and sz >= h_thresh * 0.9):
                        is_heading_block = True

            joined = " ".join(spans_text)
            if joined:
                text_parts.append(joined)
                if is_heading_block and len(joined) < 120:
                    headings.append(joined)

        pages_data.append({
            "page_num": pnum + 1,
            "text":     "\n".join(text_parts),
            "headings": headings,
        })

    doc.close()
    return pages_data


def _heading_threshold(doc) -> float:
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


def _parse_text(raw: str) -> list[dict]:
    """Plain text: split on blank lines; a line of ALL CAPS / short line acts as heading."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    pages: list[dict] = []
    for i, blk in enumerate(blocks, start=1):
        first_line = blk.splitlines()[0][:100]
        is_heading = (
            (len(first_line) <= 80 and first_line == first_line.upper() and first_line.isascii())
            or bool(re.match(r"^(Chapter|Section|Part)\b", first_line, re.I))
        )
        pages.append({
            "page_num": i,
            "text":     blk,
            "headings": [first_line] if is_heading else [],
        })
    return pages or [{"page_num": 1, "text": raw, "headings": []}]


_MD_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_SETEX_H1 = re.compile(r"^[=]{3,}\s*$")
_SETEX_H2 = re.compile(r"^[-]{3,}\s*$")
_CODE_FENCE = re.compile(r"^(`{3,}|~{3,})")


def _parse_markdown(raw: str) -> list[dict]:
    """Markdown: ATX headings (`#`), setext headings (`===`/`---`), and code fences."""
    lines = raw.splitlines()
    pages: list[dict] = []
    cur_num = 1
    cur_heading: Optional[str] = None
    cur_body: list[str] = []
    in_code_block = False
    prev_line = ""

    def flush():
        nonlocal cur_heading, cur_body
        # Skip empty flushes
        body_text = "\n".join(cur_body).strip()
        if not body_text and cur_heading is None:
            return
        pages.append({
            "page_num": cur_num,
            "text":     body_text,
            "headings": [cur_heading] if cur_heading else [],
        })

    for ln in lines:
        stripped = ln.rstrip()

        # Track fenced code blocks — headings inside code are not real headings
        fence_m = _CODE_FENCE.match(stripped) if not in_code_block else None
        if in_code_block:
            cur_body.append(ln)
            if fence_m and fence_m.group(1)[:3] == stripped.lstrip()[:3]:
                in_code_block = False
            prev_line = stripped
            continue
        if fence_m:
            in_code_block = True
            cur_body.append(ln)
            prev_line = stripped
            continue

        # ATX heading: # Heading
        m = _MD_HEAD.match(stripped)
        if m:
            flush()
            cur_num = len(pages) + 1
            level = len(m.group(1))
            cur_heading = "#" * level + " " + m.group(2).strip()
            cur_body = [stripped]
            prev_line = stripped
            continue

        # Setext heading: underline with === (H1) or --- (H2)
        if _SETEX_H1.match(stripped) and prev_line.strip():
            flush()
            cur_num = len(pages) + 1
            cur_heading = "# " + prev_line.strip()
            # Replace the body with the heading + underline
            cur_body = [prev_line, stripped]
            prev_line = stripped
            continue
        if _SETEX_H2.match(stripped) and prev_line.strip():
            flush()
            cur_num = len(pages) + 1
            cur_heading = "## " + prev_line.strip()
            cur_body = [prev_line, stripped]
            prev_line = stripped
            continue

        cur_body.append(ln)
        prev_line = stripped

    flush()
    return pages or [{"page_num": 1, "text": raw, "headings": []}]


def _parse_docx(path: str) -> list[dict]:
    """DOCX via python-docx: Heading 1/2/3 styles act as headings."""
    from docx import Document

    doc = Document(path)
    pages: list[dict] = []
    cur_num = 1
    cur_heading: Optional[str] = None
    cur_body: list[str] = []

    def flush():
        nonlocal cur_heading, cur_body
        if not cur_body and cur_heading is None:
            return
        pages.append({
            "page_num": cur_num,
            "text":     "\n".join(cur_body).strip(),
            "headings": [cur_heading] if cur_heading else [],
        })

    for para in doc.paragraphs:
        style = (para.style.name or "").lower()
        txt = para.text.strip()
        if not txt:
            continue
        if style.startswith("heading"):
            flush()
            cur_num = len(pages) + 1
            cur_heading = txt
            cur_body = [txt]
        else:
            cur_body.append(txt)
    flush()
    return pages or [{"page_num": 1, "text": "", "headings": []}]


# ──────────────────────────────────────────────────────────────
# TREE BUILDER  (with disk cache)
# ──────────────────────────────────────────────────────────────

def _compute_doc_id(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(doc_id: str) -> Path:
    return CACHE_DIR / f"{doc_id}.json"


def load_cached_tree(doc_id: str) -> Optional[DocumentTree]:
    """Hydrate a tree from the disk cache, or None if absent."""
    p = _cache_path(doc_id)
    if not p.exists():
        return None
    try:
        return DocumentTree.from_cache_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        logger.exception("Failed to read cache %s", p)
        return None


def build_document_tree(path: str, filename: str) -> DocumentTree:
    """
    Build (or load from cache) a nested DocumentTree from a local file.
    Identical file content → same doc_id → cache hit → instant load.
    """
    doc_id = _compute_doc_id(path)
    cached = load_cached_tree(doc_id)
    if cached is not None:
        # keep the uploaded display name, but reuse the parsed tree
        cached.filename = filename
        cached.from_cache = True
        logger.info("Cache HIT %s — %d sections", doc_id, len(cached.sections))
        return cached

    pages, source_format = parse_file(path)
    total_pages = len(pages)
    sections = _sections_from_pages(pages)

    tree = DocumentTree(
        filename=filename,
        total_pages=total_pages,
        sections=sections,
        doc_id=doc_id,
        source_format=source_format,
        from_cache=False,
    )
    tree.build_index()

    # persist to cache
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(doc_id).write_text(
            json.dumps(tree.to_cache_dict(), ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        logger.exception("Failed to write cache for %s", doc_id)

    logger.info("Built tree %s: %d sections from %d pages (%s)",
                doc_id, len(sections), total_pages, source_format)
    return tree


def _sections_from_pages(pages: list[dict]) -> list[Section]:
    """Group pages into nested Sections using detected headings + level nesting."""
    sections: list[Section] = []
    sid_ctr = 0

    def new_sid() -> str:
        nonlocal sid_ctr
        sid_ctr += 1
        return f"s_{sid_ctr:03d}"

    # Build a flat list of (title, level, page_start, page_end, text) first
    flat: list[dict] = []
    cur_title: Optional[str] = None
    cur_level = 3
    cur_start = pages[0]["page_num"] if pages else 1
    cur_texts: list[str] = []

    def heading_level(h: str) -> int:
        # markdown '#','##','###','####' → 1,2,3,3 ; ALL-CAPS short line → 1 ; else 2
        m = _MD_HEAD.match(h)
        if m:
            return min(len(m.group(1)), 3)
        if len(h) <= 60 and h == h.upper() and any(c.isalpha() for c in h):
            return 1
        return 2

    def flush(end_page: int):
        nonlocal cur_title, cur_level, cur_start, cur_texts
        if cur_texts or cur_title:
            title_clean = cur_title
            if title_clean:
                m = _MD_HEAD.match(title_clean)
                if m:
                    title_clean = m.group(2).strip()
            flat.append({
                "title":      title_clean or f"Page {cur_start}",
                "level":      cur_level,
                "page_start": cur_start,
                "page_end":   end_page,
                "text":       "\n".join(cur_texts),
            })

    for pd in pages:
        pnum = pd["page_num"]
        headings = pd["headings"]
        text = pd["text"]

        if headings and cur_texts:
            flush(pnum - 1)
            cur_title = headings[0]
            cur_level = heading_level(headings[0])
            cur_start = pnum
            cur_texts = [text]
        else:
            if cur_title is None:
                cur_title = headings[0] if headings else f"Page {pnum}"
                cur_level = heading_level(cur_title) if headings else 3
                cur_start = pnum
            cur_texts.append(text)

    last_page = pages[-1]["page_num"] if pages else 1
    flush(last_page)

    # Fallback: no real headings → one section per non-empty page
    if all(f["title"].startswith("Page ") for f in flat):
        flat = [{
            "title":      f"Page {pd['page_num']}",
            "level":      3,
            "page_start": pd["page_num"],
            "page_end":   pd["page_num"],
            "text":       pd["text"],
        } for pd in pages if pd["text"].strip()]

    # Convert flat list → nested Sections (stack-based parent tracking)
    for f in flat:
        sections.append(Section(
            section_id=new_sid(),
            title=f["title"],
            level=f["level"],
            page_start=f["page_start"],
            page_end=f["page_end"],
            text=f["text"],
        ))

    _link_parents(sections)
    return sections


def _link_parents(sections: list[Section]):
    """Compute parent_id / children from section levels using a stack."""
    stack: list[Section] = []   # potential parents, ordered by level
    for s in sections:
        while stack and stack[-1].level >= s.level:
            stack.pop()
        if stack:
            s.parent_id = stack[-1].section_id
            stack[-1].children.append(s.section_id)
        stack.append(s)


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
    """
    Non-streaming-style call that internally streams.
    We keep the connection alive by reading tokens as they arrive
    (stream=True) instead of blocking until the full response is ready,
    which avoids ReadTimeout on long generations.
    """
    chunks: list[str] = []
    for token in ollama_chat_stream(model, messages):
        chunks.append(token)
    return "".join(chunks)


def ollama_chat_stream(model: str, messages: list[dict]):
    """Yield text tokens from Ollama streaming API."""
    with requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        timeout=(10, None),   # (connect, read) — no read timeout while streaming
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
# STEP A — TWO-STAGE LLM TREE SEARCH
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


def _extract_id_array(raw: str, limit: int) -> Optional[list[str]]:
    """Robustly pull a JSON string-array out of an LLM response."""
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not match:
        return None
    try:
        ids = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(ids, list):
        return None
    out = [x for x in ids if isinstance(x, str)]
    return out[:limit] if out else None


def find_relevant_sections(
    tree: DocumentTree,
    query: str,
    model: str,
    top_k: int = TOP_K_SECTIONS,
) -> list[Section]:
    """
    Two-stage retrieval:
      Stage 1 — LLM reasons over top-level chapters → picks chapters
      Stage 2 — LLM drills into each chosen chapter's children → picks leaves
    Falls back to BM25-ish keyword overlap (stdlib only) if the LLM misbehaves.
    """
    roots = tree.roots_sections()

    # ── STAGE 1: pick chapters from top-level roots ──────────────
    # If a doc has many flat roots (no nesting), cap the prompt size so the
    # LLM finishes in reasonable time. Anything beyond the cap goes to Stage 2
    # as one big candidate pool instead of being reasoned over individually.
    MAX_ROOTS_IN_PROMPT = 30
    chapter_ids: Optional[list[str]] = None
    if roots:
        roots_for_prompt = roots[:MAX_ROOTS_IN_PROMPT]
        pick_n = min(top_k, len(roots_for_prompt))
        toc = [r.to_summary_dict() for r in roots_for_prompt]
        messages = [
            {"role": "system",
             "content": _TREE_SEARCH_SYS.format(top_k=pick_n)},
            {"role": "user", "content": (
                f"Document top-level chapters:\n{json.dumps(toc, indent=2)}\n\n"
                f"Question: {query}\n\n"
                f"Return up to {pick_n} chapter IDs as a JSON array."
            )},
        ]
        try:
            raw = ollama_chat(model, messages)
            chapter_ids = _extract_id_array(raw, pick_n)
        except Exception:
            logger.exception("Stage-1 tree search failed")

    # ── STAGE 2: drill into children of chosen chapters ──────────
    chosen_ids: list[str] = []
    drillable = [
        tree.get(cid) for cid in (chapter_ids or [])
        if tree.get(cid) and tree.children_of(cid)
    ]

    if drillable:
        # gather all candidate children with their parent context
        candidates = []
        for ch in drillable:
            for child in tree.children_of(ch.section_id):
                sd = child.to_summary_dict()
                sd["parent"] = ch.title[:60]
                candidates.append(sd)

        if candidates:
            messages = [
                {"role": "system",
                 "content": _TREE_SEARCH_SYS.format(top_k=top_k)},
                {"role": "user", "content": (
                    f"Candidate sub-sections (under the chosen chapters):\n"
                    f"{json.dumps(candidates, indent=2)}\n\n"
                    f"Question: {query}\n\n"
                    f"Return up to {top_k} sub-section IDs as a JSON array."
                )},
            ]
            try:
                raw = ollama_chat(model, messages)
                leaf_ids = _extract_id_array(raw, top_k)
                if leaf_ids:
                    chosen_ids = leaf_ids
            except Exception:
                logger.exception("Stage-2 tree search failed")

        # If stage 2 didn't resolve, fall back to the chapters themselves
        if not chosen_ids:
            chosen_ids = [c.section_id for c in drillable][:top_k]

    elif chapter_ids:
        # Roots had no children → use the chosen roots directly
        chosen_ids = chapter_ids[:top_k]

    # ── resolve to Section objects ───────────────────────────────
    resolved = [tree.get(sid) for sid in chosen_ids if tree.get(sid)]
    if resolved:
        return resolved

    # ── ultimate fallback: BM25-ish keyword overlap (no internet) ─
    logger.warning("LLM tree search resolved nothing — using keyword fallback")
    return _keyword_fallback(tree, query, top_k)


def _keyword_fallback(tree: DocumentTree, query: str, top_k: int) -> list[Section]:
    """Stdlib-only BM25-ish ranking over section titles+preview. Last resort."""
    stop = set("the a an of to in on for and or is are with what how why when \
                which this that it s".split())
    q_terms = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop]
    if not q_terms:
        return tree.all_sections()[:top_k]

    scored: list[tuple[float, Section]] = []
    for s in tree.all_sections():
        hay = (s.title + " " + s.text[:300]).lower()
        score = sum(hay.count(t) for t in q_terms)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored if _ > 0][:top_k]
    return best or tree.all_sections()[:top_k]


# ──────────────────────────────────────────────────────────────
# STEP B — FINAL ANSWER (streaming)
# ──────────────────────────────────────────────────────────────

_ANSWER_SYS = """\
You are a helpful assistant answering questions about a specific document.

Use ONLY the provided document sections to answer.
- Be concise and accurate.
- If the answer is not present in the sections, say so clearly.
- Always cite the page number(s), e.g. (Page 4) or (Pages 12-14).
- Format your response in clean markdown.
"""


def build_context(sections: list[Section]) -> str:
    parts = []
    for s in sections:
        pages = (f"Page {s.page_start}" if s.page_start == s.page_end
                 else f"Pages {s.page_start}-{s.page_end}")
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
    # 1 — Two-stage tree search
    relevant = find_relevant_sections(tree, query, model)
    yield ("sections", [s.to_dict() for s in relevant])

    # 2 — Build context
    context = build_context(relevant)

    # 3 — Messages
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

    # 4 — Stream answer
    for token in ollama_chat_stream(model, messages):
        yield ("token", token)

    yield ("done", "")
