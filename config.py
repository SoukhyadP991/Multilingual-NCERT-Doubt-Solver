import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectorized")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model Configuration
# User must place the GGUF model in the models directory
# Example: "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" 
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ingestion Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RAG Parameters
TOP_K_RETRIEVAL = 5
TEMPERATURE = 0.1  # Low temperature for grounded answers
MAX_NEW_TOKENS = 512
CONTEXT_WINDOW = 4096

# System Prompt
SYSTEM_PROMPT = """You are a helpful NCERT Doubt Solver for students.
Answer based ONLY on the provided Context.
Structure your answer in clear paragraphs.
Formulas and equations must be on separate lines.
Do NOT use inline citations.
Always list the Source at the very end of your response."""
