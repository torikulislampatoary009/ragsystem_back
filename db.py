"""
Database connection for Render -> Aiven MySQL.
Every credential comes from separate environment variables:
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL_CA (PEM text), DB_SSL_VERIFY
"""
import os
import tempfile
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

load_dotenv()

_ca_path = None


def _ca_file():
    """Write the PEM from DB_SSL_CA into a temp file once and return its path."""
    global _ca_path
    pem = os.getenv("DB_SSL_CA", "").strip()
    if not pem:
        return None
    if _ca_path and os.path.exists(_ca_path):
        return _ca_path
    pem = pem.replace("\\n", "\n")  # allow single-line values with literal \n
    f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".pem")
    f.write(pem)
    f.close()
    _ca_path = f.name
    return _ca_path


def get_connection():
    cfg = dict(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "defaultdb"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
        connect_timeout=15,
    )
    local = cfg["host"] in ("localhost", "127.0.0.1")
    if not local:  # Aiven requires TLS
        ca = _ca_file()
        verify = os.getenv("DB_SSL_VERIFY", "true").lower() != "false"
        if ca and verify:
            cfg["ssl"] = {"ca": ca}
        else:
            cfg["ssl"] = {"ca": ca} if ca else {}
            cfg["ssl_verify_cert"] = False
            cfg["ssl_verify_identity"] = False
    return pymysql.connect(**cfg)


def init_schema():
    conn = get_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                num_chunks INT NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id INT NOT NULL,
                chunk_index INT NOT NULL,
                page INT NULL,
                content MEDIUMTEXT NOT NULL,
                embedding MEDIUMBLOB NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id INT NULL,
                question TEXT NOT NULL,
                answer MEDIUMTEXT NOT NULL,
                sources TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
    finally:
        conn.close()
