import os
from pathlib import Path



class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Model:
        EMBEDDINGS = "BAAI/bge-small-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "gemma2:9b"
        REMOTE_LLM = None  # Set to None since no Groq key
        REMOTE_LLM_2 = "mixtral-8x7b-32768"  # Second remote LLM
        # Use Hugging Face Inference API (set HUGGINGFACEHUB_API_TOKEN in .env)
        USE_HUGGINGFACE = True
        HUGGING_FACE_MODEL = "google/flan-t5-large"  # change to preferred HF model repo_id
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000
        USE_LOCAL = False
        USE_MULTI_MODEL = True  # Enable multi-model LLM ensemble

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False
        USE_MULTIMODAL = True  # Enable multi-modal retrieval
        USE_CITATIONS = True   # Include citations in answers

    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 6
