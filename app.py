"""
Flask backend for Vectorless RAG Web UI
"""

import os
import json
import uuid
import logging
from pathlib import Path

from flask import (
    Flask, request, jsonify, render_template,
    Response, stream_with_context,
)
from werkzeug.utils import secure_filename

from rag_engine import (
    build_document_tree,
    answer_query_stream,
    list_ollama_models,
    DEFAULT_MODEL,
    DocumentTree,
)

# ──────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB

# In-memory session store  session_id → {tree, model, history}
SESSIONS: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── /api/models ────────────────────────────────────────────────
@app.route("/api/models")
def api_models():
    models = list_ollama_models()
    return jsonify({"models": models, "default": DEFAULT_MODEL})


# ── /api/upload ────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    model = request.form.get("model", DEFAULT_MODEL)

    # Save PDF
    filename  = secure_filename(f.filename)
    pdf_path  = UPLOAD_FOLDER / filename
    f.save(str(pdf_path))

    try:
        tree = build_document_tree(str(pdf_path), filename)
    except Exception as e:
        logger.exception("Failed to parse PDF")
        return jsonify({"error": f"Failed to parse PDF: {e}"}), 500

    # Create session
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "tree":    tree,
        "model":   model,
        "history": [],
    }

    section_info = [s.to_dict() for s in tree.all_sections()]
    return jsonify({
        "session_id":   session_id,
        "filename":     filename,
        "total_pages":  tree.total_pages,
        "total_sections": len(section_info),
        "sections":     section_info,
        "tree_summary": tree.tree_summary(),
    })


# ── /api/chat  (SSE streaming) ─────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    query      = (body.get("query") or "").strip()

    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    if not query:
        return jsonify({"error": "Empty query"}), 400

    sess: dict = SESSIONS[session_id]
    tree:  DocumentTree  = sess["tree"]
    model: str           = sess["model"]
    history: list[dict]  = sess["history"]

    def event_stream():
        full_answer = []
        retrieved_sections = []

        for kind, data in answer_query_stream(tree, query, model, history):
            if kind == "sections":
                retrieved_sections = data
                payload = json.dumps({"type": "sections", "data": data})
                yield f"data: {payload}\n\n"

            elif kind == "token":
                full_answer.append(data)
                payload = json.dumps({"type": "token", "data": data})
                yield f"data: {payload}\n\n"

            elif kind == "done":
                # Persist to history (last 10 turns to avoid huge context)
                answer_text = "".join(full_answer)
                history.append({"role": "user",      "content": query})
                history.append({"role": "assistant",  "content": answer_text})
                sess["history"] = history[-20:]      # keep last 10 turns

                payload = json.dumps({
                    "type": "done",
                    "sections": retrieved_sections,
                })
                yield f"data: {payload}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /api/session/<id>  — fetch session info ────────────────────
@app.route("/api/session/<session_id>")
def api_session(session_id: str):
    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found"}), 404
    sess = SESSIONS[session_id]
    tree: DocumentTree = sess["tree"]
    return jsonify({
        "filename":     tree.filename,
        "total_pages":  tree.total_pages,
        "model":        sess["model"],
        "history_len":  len(sess["history"]) // 2,
    })


# ── /api/session/<id>/clear  — wipe chat history ──────────────
@app.route("/api/session/<session_id>/clear", methods=["POST"])
def api_clear_history(session_id: str):
    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found"}), 404
    SESSIONS[session_id]["history"] = []
    return jsonify({"ok": True})


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  🌲  Vectorless RAG  —  http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
