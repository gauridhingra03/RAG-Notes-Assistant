# src/config.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    # Pehle .env / environment variable try kar (local development ke liye)
    value = os.getenv(key)
    if value:
        return value
    # Nahi mila toh Streamlit secrets try kar (Streamlit Cloud ke liye)
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None

DEFAULT_GROQ_API_KEY = _get_secret("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"

if not DEFAULT_GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing! Create your free key from https://console.groq.com")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_PAGES = 100