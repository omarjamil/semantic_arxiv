import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb # Import ChromaDB

def initialize_model():
    """Initializes the pre-trained e5-small-v2 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
    model = AutoModel.from_pretrained('intfloat/e5-small-v2')
    return tokenizer, model

def get_embeddings(texts, tokenizer, model):
    """Generates embeddings for a list of texts using the provided model."""
    prefixed_texts = ["passage: " + text for text in texts]
    
    encoded_input = tokenizer(prefixed_texts, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy() # Convert to NumPy array for ChromaDB

def mean_pooling(model_output, attention_mask):
    """Applies mean pooling to the token embeddings to get a single sentence embedding."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_search_chroma(query_text, collection, tokenizer, model, top_k=5):
    """Performs a semantic search against the ChromaDB collection."""
    # Generate embedding for the query
    query_embedding = get_embeddings(["query: " + query_text], tokenizer, model)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(), # ChromaDB expects list of lists
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'] # Request relevant info
    )
    
    formatted_results = []
    if results and results['ids']:
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "title": results['metadatas'][0][i].get('title', 'N/A'),
                "abstract": results['documents'][0][i], # The 'document' here is the combined title+abstract
                "authors": results['metadatas'][0][i].get('authors', 'N/A'),
                "categories": results['metadatas'][0][i].get('categories', 'N/A'),
                "distance": results['distances'][0][i] # Lower distance means higher similarity
            })
    return formatted_results

def load_arxiv_data_batched(filepath, batch_size, max_lines=None):
    """Loads ArXiv data from a JSON file in batches (JSON Lines format).
    
    Args:
        filepath (str): Path to the ArXiv JSON data file.
        batch_size (int): Number of documents to yield in each batch.
        max_lines (int, optional): Maximum number of lines to read from the file.
                                   If None, reads all lines. Defaults to None.
    Yields:
        pandas.DataFrame: A DataFrame containing a batch of ArXiv papers.
    """
    buffer = []
    lines_read = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if max_lines is not None and lines_read >= max_lines:
                break # Stop reading after max_lines
            
            try:
                buffer.append(json.loads(line))
                lines_read += 1 # Increment only for successfully parsed lines
            except json.JSONDecodeError as e:
                # print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}") # Uncomment for debugging
                continue # Skip to the next line

            if len(buffer) >= batch_size:
                yield pd.DataFrame(buffer)
                buffer = []
        if buffer: # Yield any remaining data in the last batch
            yield pd.DataFrame(buffer)

def get_total_arxiv_lines(filepath):
    """Counts the total number of lines in a file.
       Note: This function does not respect max_lines from load_arxiv_data_batched.
       If you want an accurate count for a subset, you'd need to pass max_lines here too.
       For a general progress bar, total file lines are usually sufficient.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f)
    except FileNotFoundError:
        print(f"Warning: File not found at {filepath}. Returning 0 lines.")
        return 0

def load_db(chroma_db_dir, collection_name="arxiv_papers"):
    """Load the ChromaDB client and collection.
       This function expects the collection to exist or throws an error,
       as creation/recreation logic is handled in the Streamlit app.
    """
    # Ensure this points to the directory where ChromaDB stores its files
    client = chromadb.PersistentClient(path=chroma_db_dir) 
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' successfully loaded.")
        return collection
    except Exception as e:
        print(f"Error loading collection '{collection_name}': {e}")
        # This will raise an error if the collection doesn't exist, which is
        # expected behavior here as population logic is external.
        raise


def populate_db(arxiv_data_filepath, collection, tokenizer, model, batch_size=1000, max_lines=None, progress_callback=None):
    """Populates the ChromaDB with ArXiv data in batches.
    
    Args:
        arxiv_data_filepath (str): Path to the ArXiv JSON data file.
        collection (chromadb.Collection): The ChromaDB collection to populate.
        tokenizer: The pre-trained tokenizer.
        model: The pre-trained model for embeddings.
        batch_size (int): Number of documents to process in each batch.
        max_lines (int, optional): Maximum number of lines to read/process from the file.
                                   If None, processes all lines. Defaults to None.
        progress_callback (callable, optional): A function to call with current progress.
                                                 Takes (processed_count, total_lines) as args.
    """
    # If max_lines is specified, the 'total_lines' for progress bar should reflect this subset
    if max_lines is not None:
        # We need to get the min of actual lines and max_lines
        actual_total_lines_in_file = get_total_arxiv_lines(arxiv_data_filepath)
        total_lines_for_progress = min(max_lines, actual_total_lines_in_file)
    else:
        total_lines_for_progress = get_total_arxiv_lines(arxiv_data_filepath)

    processed_count = 0

    print(f"Starting database population for up to {total_lines_for_progress} papers...")

    # Pass max_lines to the batched data loader
    for batch_df in load_arxiv_data_batched(arxiv_data_filepath, batch_size, max_lines=max_lines):
        # Handle potential missing 'title' or 'abstract' keys
        batch_df['title'] = batch_df['title'].fillna('')
        batch_df['abstract'] = batch_df['abstract'].fillna('')
        batch_df['text_to_embed'] = batch_df['title'] + " " + batch_df['abstract']

        batch_embeddings = get_embeddings(batch_df['text_to_embed'].tolist(), tokenizer, model)
        
        documents = batch_df['text_to_embed'].tolist()
        metadatas = batch_df[['id', 'title', 'authors', 'categories']].to_dict(orient='records')
        ids = [str(x) for x in batch_df['id'].tolist()]

        try:
            collection.upsert( 
                embeddings=batch_embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error upserting batch to ChromaDB: {e}")
            raise 
        
        processed_count += len(batch_df)
        if progress_callback:
            progress_callback(processed_count, total_lines_for_progress) # Pass the subset total
        else:
            print(f"Processed {processed_count}/{total_lines_for_progress} papers.")

# --- REMOVE THIS BLOCK (or comment out fully) ---
# if __name__ == "__main__":
#     arxiv_data_filepath = 'arxiv-metadata-oai-snapshot.json'
#     chroma_db_path = "./chroma_arxiv_db/chroma.sqlite3" 
#     collection = load_db(chroma_db_path)
#     print(f"Loading data from {arxiv_data_filepath}...")
#     tokenizer, model = initialize_model()
#     print("Model and tokenizer loaded.")
#     print("Database populated with ArXiv data.")
#     print("Starting semantic search...")
#     search_query = "new methods"
#     print(f"\nSearching ChromaDB for: '{search_query}'")
#     search_results = semantic_search_chroma(search_query, collection, tokenizer, model, top_k=10)
#     for i, result in enumerate(search_results):
#         print(f"\n--- Result {i+1} ---")
#         print(f"ArXiv ID: {result['id']}")
#         print(f"Title: {result['title']}")
#         print(f"Combined Text: {result['abstract'][:200]}...")
#         print(f"Authors: {result['authors']}")
#         print(f"Categories: {result['categories']}")
#         print(f"Distance (lower is better): {result['distance']:.4f}")
#     search_query_2 = "impact of computers"
#     print(f"\nSearching ChromaDB for: '{search_query_2}'")
#     search_results_2 = semantic_search_chroma(search_query_2, collection, tokenizer, model, top_k=2)
#     for i, result in enumerate(search_results_2):
#         print(f"\n--- Result {i+1} ---")
#         print(f"ArXiv ID: {result['id']}")
#         print(f"Title: {result['title']}")
#         print(f"Combined Text: {result['abstract'][:200]}...")
#         print(f"Authors: {result['authors']}")
#         print(f"Categories: {result['categories']}")
#         print(f"Distance (lower is better): {result['distance']:.4f}")