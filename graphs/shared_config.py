"""
Shared configuration for LLM and embedding instances across the mental health assistant.
"""

import os
from pathlib import Path
import dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

# Model configurations
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Shared LLM instance
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3,
    base_url="http://localhost:11434"  # Default Ollama port
)

# Alternative OpenAI LLM (commented out but available)
# llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.3)

# Shared embedding instance
embed = OpenAIEmbeddings(model="text-embedding-3-small") 