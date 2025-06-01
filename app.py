import streamlit as st
import functions # Assuming your functions are in functions.py
import os # To handle file paths
import chromadb
import json

st.set_page_config(page_title="ArXiv Semantic Search", layout="centered")

# --- Configuration ---
ARXIV_DATA_FILEPATH = 'arxiv-metadata-oai-snapshot.json'
# ChromaDB stores data in a directory, not directly in .sqlite3 file.
# The .sqlite3 file is one of the files *inside* this directory.
CHROMA_DB_DIR = "./chroma_arxiv_db_test_subset" # Use a different DB directory for testing
COLLECTION_NAME = "arxiv_papers_test" # Consistent with your populate_db logic

# --- Initialize Session State for Model, Tokenizer, and Collection ---
# This prevents reloading the model/tokenizer/DB on every rerun of the script
# (e.g., when a user types in the text input).
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer, st.session_state.model = functions.initialize_model()
    st.session_state.initialized_model = True
    st.write("Tokenizer and Model loaded!")

if 'collection' not in st.session_state:
    # Initialize ChromaDB client (persistent)
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        st.session_state.collection = client.get_collection(name=COLLECTION_NAME)
        st.session_state.chroma_loaded = True
        st.write(f"ChromaDB collection '{COLLECTION_NAME}' loaded from {CHROMA_DB_DIR}.")
    except Exception as e:
        st.error(f"Error loading ChromaDB collection: {e}")
        st.write("Please ensure the database has been populated by running `populate_vector_db` in a separate script or directly below.")
        st.session_state.collection = None # Set to None if loading fails

# --- Function to populate DB (for initial setup or re-population) ---
# @st.cache_resource # Cache the function result to prevent re-running on every page load
# def populate_vector_db_streamlit(arxiv_data_filepath, chroma_db_dir, collection_name, _tokenizer, _model):
#     """Populate the vector database with ArXiv data.
#        This version handles Streamlit's caching and UI updates.
#     """
#     st.info("Populating database... This may take a while depending on data size.")
#     try:
#         client = chromadb.PersistentClient(path=chroma_db_dir)
#         # Attempt to get the collection, if it exists, delete and recreate to ensure clean populate
#         try:
#             client.delete_collection(name=collection_name)
#             st.warning(f"Existing collection '{collection_name}' deleted for re-population.")
#         except Exception as e:
#             st.info(f"No existing collection '{collection_name}' found to delete or other issue: {e}")

#         collection = client.create_collection(name=collection_name)
        
#         # Load and process data in batches to be memory efficient for large datasets
#         # You might need to adjust batch_size based on your system's memory
#         batch_size = 1000
#         processed_count = 0

#         # Generator to yield data in batches
#         def batch_data_generator(filepath, batch_size):
#             buffer = []
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     buffer.append(json.loads(line))
#                     if len(buffer) >= batch_size:
#                         yield pd.DataFrame(buffer)
#                         buffer = []
#                 if buffer: # Yield any remaining data
#                     yield pd.DataFrame(buffer)

#         # Iterate through batches and add to ChromaDB
#         for batch_df in functions.load_arxiv_data_batched(arxiv_data_filepath, batch_size): # Assuming you add this to functions.py
#             batch_df['text_to_embed'] = batch_df['title'] + " " + batch_df['abstract']
#             batch_embeddings = functions.get_embeddings(batch_df['text_to_embed'].tolist(), _tokenizer, _model)
            
#             # Prepare data for ChromaDB
#             documents = batch_df['text_to_embed'].tolist()
#             metadatas = batch_df[['id', 'title', 'authors', 'categories']].to_dict(orient='records')
#             ids = [str(x) for x in batch_df['id'].tolist()]

#             collection.add(
#                 embeddings=batch_embeddings.tolist(),
#                 documents=documents,
#                 metadatas=metadatas,
#                 ids=ids
#             )
#             processed_count += len(batch_df)
#             st.progress(processed_count / functions.get_total_arxiv_lines(arxiv_data_filepath), text=f"Processing {processed_count} papers...") # Requires a way to get total lines
            
#         st.success(f"Database populated with {processed_count} ArXiv papers!")
#         st.session_state.collection = collection # Update session state with the new collection
#         st.session_state.chroma_loaded = True
#         return collection
#     except Exception as e:
#         st.error(f"Failed to populate database: {e}")
#         st.session_state.chroma_loaded = False
#         return None

# --- Streamlit UI ---
st.title("📚 ArXiv Semantic Search")
st.markdown("Search ArXiv papers semantically using `e5-small-v2` embeddings and ChromaDB.")

# Option to (re)populate the database
# if st.sidebar.button("Populate/Re-populate Database"):
#     # Clear cache before populating to ensure fresh data
#     st.cache_resource.clear()
#     st.session_state.collection = populate_vector_db_streamlit(
#         ARXIV_DATA_FILEPATH, CHROMA_DB_DIR, COLLECTION_NAME, 
#         st.session_state.tokenizer, st.session_state.model
#     )

# Check if DB is loaded before allowing search
if st.session_state.collection is None:
    st.warning("Database not yet loaded or populated. Please click 'Populate/Re-populate Database' or ensure previous run was successful.")
else:
    search_query = st.text_input(
        "Enter your search query:", 
        placeholder="e.g., quantum computing in cryptography"
    )

    top_k_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        if search_query:
            with st.spinner("Searching..."):
                search_results = functions.semantic_search_chroma(
                    search_query, 
                    st.session_state.collection, 
                    st.session_state.tokenizer, 
                    st.session_state.model, 
                    top_k=top_k_results
                )
            
            if search_results:
                st.subheader(f"Top {len(search_results)} Results for '{search_query}':")
                for i, result in enumerate(search_results):
                    st.markdown(f"---")
                    st.markdown(f"**{i+1}. [{result['title']}](https://arxiv.org/abs/{result['id']})**")
                    st.write(f"**ArXiv ID:** `{result['id']}`")
                    st.write(f"**Authors:** {result['authors']}")
                    st.write(f"**Categories:** `{result['categories']}`")
                    st.write(f"**Similarity Distance (lower is better):** `{result['distance']:.4f}`")
                    
                    with st.expander("Show Abstract"):
                        # Ensure 'document' contains the full abstract if it's what you want to display
                        # If 'document' is combined title+abstract, you might need to store original abstract in metadata
                        st.write(result['abstract']) 
            else:
                st.info("No results found for your query.")
        else:
            st.warning("Please enter a search query.")

st.sidebar.subheader("About")
st.sidebar.info(
    "This is a prototype semantic search tool for ArXiv papers built with "
    "`e5-small-v2` for embeddings and `ChromaDB` for vector storage, "
    "powered by `Streamlit`."
)