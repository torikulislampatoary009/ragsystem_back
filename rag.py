"""RAG core: PDF/text -> chunks -> embeddings -> FAISS (LangChain) -> open-source LLM."""
import io
import os
import json
import numpy as np
from typing import List

from pypdf import PdfReader
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

import db

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
CHUNK_SIZE, CHUNK_OVERLAP = 800, 120
TOP_K = 4
MIN_SCORE = 0.35  # cosine similarity below this => "not in the document"

NOT_FOUND = "I could not find the answer to that in the uploaded document."

PROMPT = """You are a document assistant. Answer the question STRICTLY and ONLY using the context below.
If the context does not contain the answer, reply exactly: "{nf}"
Do not use outside knowledge. Be concise.

Context:
{context}

Question: {question}
Answer:"""


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible wrapper around fastembed (ONNX, small RAM footprint)."""
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            cls._model = TextEmbedding(model_name=EMBED_MODEL)
        return cls._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.model().embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self.model().embed([text]))).tolist()


embeddings = FastEmbedEmbeddings()
_store = None  # in-memory FAISS, rebuilt from MySQL


# ---------- loading & indexing ----------
def extract_pages(filename: str, data: bytes):
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, data.decode("utf-8", errors="ignore"))]


def ingest(filename: str, data: bytes) -> dict:
    pages = extract_pages(filename, data)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append((page, piece.strip()))
    if not chunks:
        raise ValueError("No readable text found (scanned PDFs need OCR).")

    vecs = np.array(embeddings.embed_documents([c[1] for c in chunks]), dtype="float32")
    conn = db.get_connection()
    try:
        with conn.cursor() as c:
            c.execute("INSERT INTO documents (filename, num_chunks) VALUES (%s,%s)", (filename, len(chunks)))
            doc_id = c.lastrowid
            c.executemany(
                "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) VALUES (%s,%s,%s,%s,%s)",
                [(doc_id, i, ch[0], ch[1], vecs[i].tobytes()) for i, ch in enumerate(chunks)],
            )
    finally:
        conn.close()
    rebuild_store()
    return {"id": doc_id, "filename": filename, "chunks": len(chunks)}


def rebuild_store():
    """Load every chunk from MySQL into a LangChain FAISS store (cosine via inner product)."""
    global _store
    conn = db.get_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT document_id, chunk_index, page, content, embedding FROM chunks")
            rows = c.fetchall()
    finally:
        conn.close()
    if not rows:
        _store = None
        return
    pairs = [(r["content"], np.frombuffer(r["embedding"], dtype="float32").tolist()) for r in rows]
    metas = [{"document_id": r["document_id"], "page": r["page"], "chunk": r["chunk_index"]} for r in rows]
    _store = FAISS.from_embeddings(pairs, embeddings, metadatas=metas,
                                   distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                                   normalize_L2=True)


# ---------- answering ----------
def retrieve(question: str, document_id=None):
    if _store is None:
        return []
    kwargs = {"k": TOP_K * (4 if document_id else 1)}
    hits = _store.similarity_search_with_score(question, **kwargs)
    out = []
    for doc, score in hits:
        if document_id and doc.metadata["document_id"] != int(document_id):
            continue
        if score >= MIN_SCORE:
            out.append((doc, float(score)))
    return out[:TOP_K]


def generate(question: str, context: str) -> str:
    token = os.getenv("HF_API_TOKEN")
    if not token:
        return None
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=HF_MODEL, token=token, timeout=60)
    prompt = PROMPT.format(nf=NOT_FOUND, context=context, question=question)
    res = client.chat_completion(messages=[{"role": "user", "content": prompt}],
                                 max_tokens=400, temperature=0.1)
    return res.choices[0].message.content.strip()


def answer(question: str, document_id=None) -> dict:
    hits = retrieve(question, document_id)
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}
    context = "\n\n---\n\n".join(d.page_content for d, _ in hits)
    sources = [{"page": d.metadata["page"], "score": round(s, 3), "text": d.page_content[:300]} for d, s in hits]
    try:
        text = generate(question, context)
    except Exception as e:  # LLM busy / model cold start
        text = None
        err = str(e)[:120]
    if text is None:
        text = "(LLM unavailable - showing the most relevant passages)\n\n" + hits[0][0].page_content
    return {"answer": text, "sources": sources}
