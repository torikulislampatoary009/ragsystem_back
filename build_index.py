"""
Build the RAG 'model file' offline (the FAISS index) from a PDF/TXT.
    python build_index.py path/to/company_wiki.pdf
Creates:  model/rag_index.faiss   model/rag_chunks.json
(The deployed app also stores chunks+vectors in MySQL so they survive Render restarts;
 this script is for local testing / committing a ready-made index to GitHub.)
"""
import sys, json
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import extract_pages, embeddings, CHUNK_SIZE, CHUNK_OVERLAP

path = sys.argv[1]
pages = extract_pages(path, open(path, "rb").read())
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = [{"page": p, "text": t.strip()} for p, txt in pages for t in splitter.split_text(txt) if t.strip()]

vecs = np.array(embeddings.embed_documents([c["text"] for c in chunks]), dtype="float32")
faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, "model/rag_index.faiss")
json.dump(chunks, open("model/rag_chunks.json", "w"), ensure_ascii=False, indent=1)
print(f"Saved {len(chunks)} chunks -> model/rag_index.faiss")
