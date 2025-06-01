import functions
import os
import chromadb

# --- Configuration ---
ARXIV_DATA_FILEPATH = 'arxiv-metadata-oai-snapshot.json'
CHROMA_DB_DIR = "./chroma_arxiv_db_test_subset" # Use a different DB directory for testing
COLLECTION_NAME = "arxiv_papers_test" # Use a different collection name for testing

# --- For testing with a small subset ---
TEST_MAX_LINES = 100 # Process only the first 100 entries

def main():
    print("--- Starting Standalone Test (Subset) ---")

    # --- 1. Initialize Model and Tokenizer ---
    print("\n1. Initializing E5-small-v2 model and tokenizer...")
    tokenizer, model = functions.initialize_model()
    print("   Model and tokenizer loaded.")

    # --- 2. Set up ChromaDB Client and Collection ---
    print("\n2. Setting up ChromaDB client for subset testing...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    print(f"   ChromaDB client initialized. Data path: {CHROMA_DB_DIR}")

    # --- 3. Populate or Re-populate the Database (Subset) ---
    print(f"   Attempting to delete existing collection '{COLLECTION_NAME}' if it exists...")
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"   Collection '{COLLECTION_NAME}' deleted successfully.")
    except Exception as e:
        print(f"   Collection '{COLLECTION_NAME}' did not exist or could not be deleted: {e}")
    
    print(f"   Creating new collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"   Collection '{COLLECTION_NAME}' created.")

    print(f"\n3. Populating ChromaDB with first {TEST_MAX_LINES} ArXiv data entries...")
    def console_progress_callback(processed_count, total_lines):
        if total_lines > 0:
            percentage = (processed_count / total_lines) * 100
            print(f"   Processed {processed_count}/{total_lines} papers ({percentage:.2f}%)", end='\r')
        else:
            print(f"   Processed {processed_count} papers.", end='\r')

    try:
        functions.populate_db(
            ARXIV_DATA_FILEPATH, 
            collection, 
            tokenizer, 
            model, 
            max_lines=TEST_MAX_LINES, # <--- Pass the max_lines here!
            progress_callback=console_progress_callback
        )
        print("\n   Database population complete for subset.")
    except Exception as e:
        print(f"\nAn error occurred during subset database population: {e}")
        # Handle error as needed, e.g., exit or set collection to None
        return

    # --- 4. Perform Semantic Searches ---
    print("\n4. Performing semantic searches on the subset...")

    search_queries = [
        "semantic search improvements", # Query relevant to a subset of data
        "brief history of natural language processing",
        "quantum entanglement" # May or may not be in the first 100, good for testing empty results
    ]

    for query in search_queries:
        print(f"\n--- Searching for: '{query}' ---")
        search_results = functions.semantic_search_chroma(query, collection, tokenizer, model, top_k=3)

        if search_results:
            for i, result in enumerate(search_results):
                print(f"  Result {i+1} (Distance: {result['distance']:.4f}):")
                print(f"    ID: {result['id']}")
                print(f"    Title: {result['title']}")
                print(f"    Authors: {result['authors']}")
                print(f"    Categories: {result['categories']}")
                print(f"    Abstract (excerpt): {result['abstract'][:150]}...")
        else:
            print(f"  No results found for '{query}'.")

    print("\n--- Standalone Test (Subset) Complete ---")

if __name__ == "__main__":
    main()