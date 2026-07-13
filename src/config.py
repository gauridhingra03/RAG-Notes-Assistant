# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing! Create your free key from https://console.groq.com")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_PAGES = 100