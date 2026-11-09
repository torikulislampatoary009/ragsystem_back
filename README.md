# RAG Backend (Flask + LangChain + FAISS + MySQL)

## Render (free) setup
- New Web Service -> connect your GitHub repo (root dir = this folder)
- Build: `pip install -r requirements.txt`   Start: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
- Environment variables (add each separately):
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL_CA (paste Aiven CA PEM),
  HF_API_TOKEN, HF_MODEL (optional), CORS_ORIGINS (your frontend URL)
- Python version: set env var PYTHON_VERSION=3.11.9

## Aiven (free MySQL)
Copy Host, Port, User, Password, DB name and the CA certificate from the service overview page.
Tables are created automatically on the first request.

## Local test
cp .env.example .env  (fill in) -> pip install -r requirements.txt -> python app.py

## Notes
- Render free has 512 MB RAM and sleeps after 15 min idle: first request takes ~50 s.
- Vectors are stored in MySQL, so uploads survive Render restarts (its disk is ephemeral).
- `python build_index.py doc.pdf` creates model/rag_index.faiss + rag_chunks.json (the model file).
