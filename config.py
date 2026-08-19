from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================================
# Environment
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ============================================================
# Paths
# ============================================================
PDF_DIR = BASE_DIR / "data"
CHUNKS_FILE = BASE_DIR / "chunks.json"

# ============================================================
# Chunking Configuration
# ============================================================
MIN_TOKENS = 100
MAX_TOKENS = 700
OVERLAP_TOKENS = 100

# ============================================================
# Hugging Face (Vision & Embeddings)
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

# 1. Vision Model (For OCR)
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "google/gemma-4-26B-A4B-it")

# 2. Embedding Model (For Vector generation in Supabase)
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIMENSION = 768

# ============================================================
# Groq (LLM for Graph Extraction & Generation)
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# ============================================================
# OCR Configuration
# ============================================================
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_TEXT_THRESHOLD = int(os.getenv("OCR_TEXT_THRESHOLD", "50"))
OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "1024"))

# ============================================================
# Supabase Configuration (Vector Database)
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ============================================================
# Neo4j Configuration (Knowledge Graph)
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ============================================================
# Graph RAG Retrieval Limits
# ============================================================
TOP_K = 3
GRAPH_TOP_K = 5
RAG_MAX_TOKENS = 2048
RAG_TEMPERATURE = 0.0

# ============================================================
# Validation
# ============================================================
def validate_config():
    """
    Validate required environment variables
    before running the pipeline.
    """
    required = {
        "HF_TOKEN": HF_TOKEN,
        "GROQ_API_KEY": GROQ_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

if __name__ == "__main__":
    validate_config()
    print("✅ Config validated successfully. All required keys are present.")