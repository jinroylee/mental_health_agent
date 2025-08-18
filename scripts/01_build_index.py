import os, glob, dotenv, sys
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time

# Import shared embedding instance
sys.path.append(str(Path(__file__).resolve().parents[1]))
from graphs.shared_config import embed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR   = PROJECT_ROOT / "data_raw"
CHUNK_DIR  = PROJECT_ROOT / "data_chunks"     # optional cache

# Default chunk size and overlap
DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP = 1200, 150

# Chunk sizes per subdirectory - you can customize these
CHUNK_SIZES = {
    "counseling_resource": {"chunk_size": 1500, "overlap": 250},
    "crisis_resource": {"chunk_size": 1500, "overlap": 250},      # Smaller chunks for crisis info
    "reframe_template": {"chunk_size": 1500, "overlap": 250},     # Smaller for templates
    "therapy_resource": {"chunk_size": 2000, "overlap": 300},    # Larger for comprehensive therapy docs
}

INDEX_NAME = os.environ.get("INDEX_NAME")

def load_and_chunk_documents():
    """Load documents from subdirectories, add doc_type metadata, and chunk with appropriate sizes."""
    all_chunks = []
    # Define the expected subdirectories
    subdirs = ["counseling_resource", "crisis_resource", "reframe_template", "therapy_resource"]
    
    for subdir in subdirs:
        subdir_path = DATA_DIR / subdir
        if not subdir_path.exists():
            print(f"Warning: Subdirectory {subdir} does not exist")
            continue
            
        print(f"Processing {subdir}...")
        subdir_docs = []
        
        # Get chunk settings for this subdirectory
        chunk_config = CHUNK_SIZES.get(subdir, {"chunk_size": DEFAULT_CHUNK_SIZE, "overlap": DEFAULT_OVERLAP})
        chunk_size = chunk_config["chunk_size"]
        overlap = chunk_config["overlap"]
        print(f"  Using chunk_size={chunk_size}, overlap={overlap}")
        
        # Process PDF files
        for pdf_file in subdir_path.glob("*.pdf"):
            try:
                loaded_docs = PyPDFLoader(str(pdf_file)).load()
                for doc in loaded_docs:
                    # Add doc_type to metadata
                    doc.metadata["doc_type"] = subdir
                subdir_docs.extend(loaded_docs)
                print(f"  Loaded PDF: {pdf_file.name}")
            except Exception as e:
                print(f"  Error loading PDF {pdf_file.name}: {e}")
        
        # Process TXT files
        for txt_file in subdir_path.glob("*.txt"):
            try:
                loaded_docs = TextLoader(str(txt_file)).load()
                for doc in loaded_docs:
                    # Add doc_type to metadata
                    doc.metadata["doc_type"] = subdir
                subdir_docs.extend(loaded_docs)
                print(f"  Loaded TXT: {txt_file.name}")
            except Exception as e:
                print(f"  Error loading TXT {txt_file.name}: {e}")
        
        # Process textClipping files
        for clip_file in subdir_path.glob("*.textClipping"):
            try:
                loaded_docs = TextLoader(str(clip_file)).load()
                for doc in loaded_docs:
                    # Add doc_type to metadata
                    doc.metadata["doc_type"] = subdir
                subdir_docs.extend(loaded_docs)
                print(f"  Loaded textClipping: {clip_file.name}")
            except Exception as e:
                print(f"  Error loading textClipping {clip_file.name}: {e}")
        
        print(f"Loaded {len(subdir_docs)} docs from {subdir}")
        
        # Chunk documents for this subdirectory with its specific settings
        if subdir_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            subdir_chunks = splitter.split_documents(subdir_docs)
            all_chunks.extend(subdir_chunks)
            print(f"Created {len(subdir_chunks)} chunks from {subdir}")
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def save_chunks(chunks):
    """Save chunks to data_chunks directory organized by doc_type."""
    CHUNK_DIR.mkdir(exist_ok=True)
    
    # Clear existing subdirectories and files
    for existing_subdir in CHUNK_DIR.iterdir():
        if existing_subdir.is_dir():
            for file in existing_subdir.glob("*.json"):
                file.unlink()
            try:
                existing_subdir.rmdir()
            except OSError:
                pass  # Directory not empty, that's okay
    
    # Group chunks by doc_type
    chunks_by_type = {}
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        if doc_type not in chunks_by_type:
            chunks_by_type[doc_type] = []
        chunks_by_type[doc_type].append(chunk)
    
    total_saved = 0
    for doc_type, type_chunks in chunks_by_type.items():
        # Create subdirectory for this doc_type
        type_dir = CHUNK_DIR / doc_type
        type_dir.mkdir(exist_ok=True)
        
        # Save chunks for this doc_type
        for i, chunk in enumerate(type_chunks):
            chunk_data = {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            
            chunk_file = type_dir / f"chunk_{i:04d}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        total_saved += len(type_chunks)
        print(f"Saved {len(type_chunks)} chunks to {type_dir}")
    
    print(f"Saved {total_saved} total chunks to {CHUNK_DIR}")

def main():
    print("Loading, chunking & embedding ...")
    chunks = load_and_chunk_documents()
    print(f"Produced {len(chunks)} chunks")

    # Save chunks to data_chunks directory
    save_chunks(chunks)

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

if __name__ == "__main__":
    sys.exit(main())
