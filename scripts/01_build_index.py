#!/usr/bin/env python
"""
Embeds all files in data_raw/ and builds a Pinecone (or Chroma) index.
Run once after you add or change documents.
"""

import os, glob, dotenv, sys
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR   = PROJECT_ROOT / "data_raw"
CHUNK_DIR  = PROJECT_ROOT / "data_chunks"     # optional cache
CHUNK_SIZE, OVERLAP = 800, 200
INDEX_NAME = os.environ.get("INDEX_NAME")

def load_documents():
    docs = []
    for fp in glob.glob(str(DATA_DIR / "*")):
        if fp.lower().endswith(".pdf"):
            docs += PyPDFLoader(fp).load()
        else:
            docs += UnstructuredFileLoader(fp).load()
    print(f"Loaded {len(docs)} raw docs")
    return docs

def save_chunks(chunks):
    """Save chunks to data_chunks directory as JSON files."""
    CHUNK_DIR.mkdir(exist_ok=True)
    
    # Clear existing chunk files
    for existing_file in CHUNK_DIR.glob("chunk_*.json"):
        existing_file.unlink()
    
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        
        chunk_file = CHUNK_DIR / f"chunk_{i:04d}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {CHUNK_DIR}")

def main():
    print("Splitting & embedding ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    chunks   = splitter.split_documents(load_documents())
    print(f"Produced {len(chunks)} chunks")

    # Save chunks to data_chunks directory
    save_chunks(chunks)

    embed = OpenAIEmbeddings(model="text-embedding-ada-002")

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    print(existing_indexes)
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="enabled",  # Defaults to "disabled"
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embed)
    vector_store.add_documents(chunks)
    
    print(vector_store.similarity_search("What is the capital of France?", k=2))

if __name__ == "__main__":
    sys.exit(main())
