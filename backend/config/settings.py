import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

BACKEND_DIR = PROJECT_ROOT / "backend"
VECTOR_STORE_DIR = BACKEND_DIR / "vector_store"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv(ENV_FILE)


# ---------------------------------------------------------
# OpenAI configuration
# ---------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

OPENAI_CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-4.1-mini",
)


# ---------------------------------------------------------
# Neo4j configuration
# ---------------------------------------------------------

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://localhost:7687",
)

NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME",
    "neo4j",
)

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ---------------------------------------------------------
# Create required directories
# ---------------------------------------------------------

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)