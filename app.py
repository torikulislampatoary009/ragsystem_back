import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import db
import rag

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ORIGINS", "*").split(","))
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

_ready = False


def ensure_ready():
    global _ready
    if not _ready:
        db.init_schema()
        rag.rebuild_store()
        _ready = True


@app.before_request
def _boot():
    if request.path != "/api/health":
        ensure_ready()


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file provided"), 400
    if not f.filename.lower().endswith((".pdf", ".txt", ".md")):
        return jsonify(error="Only PDF, TXT or MD files are supported"), 400
    try:
        return jsonify(rag.ingest(f.filename, f.read())), 201
    except ValueError as e:
        return jsonify(error=str(e)), 422


@app.get("/api/documents")
def documents():
    conn = db.get_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, filename, num_chunks, created_at FROM documents ORDER BY id DESC")
            rows = c.fetchall()
    finally:
        conn.close()
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return jsonify(rows)


@app.delete("/api/documents/<int:doc_id>")
def delete_document(doc_id):
    conn = db.get_connection()
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
    finally:
        conn.close()
    rag.rebuild_store()
    return jsonify(deleted=doc_id)


@app.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}
    q = (body.get("question") or "").strip()
    if not q:
        return jsonify(error="Question is required"), 400
    doc_id = body.get("document_id")
    result = rag.answer(q, doc_id)
    conn = db.get_connection()
    try:
        with conn.cursor() as c:
            c.execute("INSERT INTO chat_history (document_id, question, answer, sources) VALUES (%s,%s,%s,%s)",
                      (doc_id, q, result["answer"], json.dumps(result["sources"])))
    finally:
        conn.close()
    return jsonify(result)


@app.get("/api/history")
def history():
    conn = db.get_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, question, answer, created_at FROM chat_history ORDER BY id DESC LIMIT 30")
            rows = c.fetchall()
    finally:
        conn.close()
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return jsonify(rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
