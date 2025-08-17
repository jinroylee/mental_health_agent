#!/usr/bin/env python
"""
Test similarity searches on the created Pinecone index.
Supports filtering by doc_type and interactive querying.
"""

import os
import sys
import dotenv
from pathlib import Path
import json
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Import shared embedding instance
sys.path.append(str(Path(__file__).resolve().parents[1]))
from graphs.shared_config import embed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(PROJECT_ROOT / ".env")

INDEX_NAME = os.environ.get("INDEX_NAME")

# Available doc types
DOC_TYPES = ["counseling_resource", "crisis_resource", "reframe_template", "therapy_resource"]

def initialize_vector_store():
    """Initialize connection to Pinecone vector store."""
    try:
        # embed = OpenAIEmbeddings(model="text-embedding-ada-002") # This line is removed
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            print(f"Error: Index '{INDEX_NAME}' does not exist. Please run 01_build_index.py first.")
            return None
        
        index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(index=index, embedding=embed)
        
        # Get index stats
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"✅ Connected to index '{INDEX_NAME}' with {total_vectors} vectors")
        
        return vector_store
    except Exception as e:
        print(f"❌ Error connecting to vector store: {e}")
        return None

def display_results(results, query, doc_type_filter=None):
    """Display search results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"🔍 Query: '{query}'")
    if doc_type_filter:
        print(f"📂 Filter: doc_type = '{doc_type_filter}'")
    print(f"📊 Found {len(results)} results")
    print('='*60)
    
    for i, result in enumerate(results, 1):
        metadata = result.metadata
        doc_type = metadata.get('doc_type', 'unknown')
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', 'N/A')
        
        print(f"\n🔸 Result {i}:")
        print(f"  📁 Doc Type: {doc_type}")
        print(f"  📄 Source: {Path(source).name if source != 'unknown' else 'unknown'}")
        print(f"  📃 Page: {page}")
        print(f"  📝 Content Preview: {result.page_content[:200]}...")
        if len(result.page_content) > 200:
            print(f"     ... (content continues for {len(result.page_content)} total characters)")

def run_predefined_tests(vector_store):
    """Run a set of predefined test queries."""
    test_queries = [
        {
            "query": "anxiety management techniques",
            "description": "General anxiety help",
            "filter": None
        },
        {
            "query": "breathing exercises for panic attacks",
            "description": "Crisis intervention techniques",
            "filter": "crisis_resource"
        },
        {
            "query": "cognitive behavioral therapy techniques",
            "description": "CBT methods",
            "filter": "therapy_resource"
        },
        {
            "query": "how to reframe negative thoughts",
            "description": "Cognitive reframing",
            "filter": "reframe_template"
        },
        {
            "query": "signs of depression",
            "description": "Depression identification",
            "filter": "counseling_resource"
        }
    ]
    
    print("\n🧪 Running predefined test queries...")
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*40} Test {i} {'='*40}")
        print(f"Description: {test['description']}")
        
        try:
            if test["filter"]:
                results = vector_store.similarity_search(
                    test["query"], 
                    k=3, 
                    filter={"doc_type": test["filter"]}
                )
            else:
                results = vector_store.similarity_search(test["query"], k=3)
            
            display_results(results, test["query"], test["filter"])
            
        except Exception as e:
            print(f"❌ Error in test query: {e}")
    
    print(f"\n{'='*90}")
    print("✅ Predefined tests completed!")

def interactive_search(vector_store):
    """Interactive similarity search interface."""
    print("\n🔍 Interactive Similarity Search")
    print("Commands:")
    print("  - Type your query to search")
    print("  - 'filter <doc_type>' to set a filter (e.g., 'filter crisis_resource')")
    print("  - 'clear filter' to remove filter")
    print("  - 'set k <number>' to change number of results (e.g., 'set k 5')")
    print("  - 'types' to see available doc types")
    print("  - 'quit' to exit")
    
    current_filter = None
    k_value = 3
    
    while True:
        # Show current settings
        filter_display = f" [Filter: {current_filter}]" if current_filter else ""
        prompt = f"\nSearch{filter_display} [k={k_value}]: "
        
        user_input = input(prompt).strip()
        
        if user_input.lower() in {"quit", "exit"}:
            break
        elif user_input.lower() == "types":
            print(f"Available doc types: {', '.join(DOC_TYPES)}")
            continue
        elif user_input.lower().startswith("filter "):
            doc_type = user_input[7:].strip()
            if doc_type in DOC_TYPES:
                current_filter = doc_type
                print(f"✅ Filter set to: {doc_type}")
            else:
                print(f"❌ Invalid doc type. Available types: {', '.join(DOC_TYPES)}")
            continue
        elif user_input.lower() == "clear filter":
            current_filter = None
            print("✅ Filter cleared")
            continue
        elif user_input.lower().startswith("set k "):
            try:
                new_k = int(user_input[6:].strip())
                if 1 <= new_k <= 20:
                    k_value = new_k
                    print(f"✅ k value set to: {k_value}")
                else:
                    print("❌ k value must be between 1 and 20")
            except ValueError:
                print("❌ Invalid k value. Please enter a number.")
            continue
        elif not user_input:
            continue
        
        # Perform search
        try:
            if current_filter:
                results = vector_store.similarity_search(
                    user_input, 
                    k=k_value, 
                    filter={"doc_type": current_filter}
                )
            else:
                results = vector_store.similarity_search(user_input, k=k_value)
            
            display_results(results, user_input, current_filter)
            
        except Exception as e:
            print(f"❌ Search error: {e}")

def main():
    print("🧠 Mental Health Agent - Similarity Search Tester")
    print("="*50)
    
    # Initialize vector store
    vector_store = initialize_vector_store()
    if not vector_store:
        return 1
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Run predefined test queries")
        print("2. Interactive search")
        print("3. Quit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            run_predefined_tests(vector_store)
        elif choice == "2":
            interactive_search(vector_store)
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 