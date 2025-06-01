# Semantic Search of ArXiv papers
Carry out semantic search of arxiv papers [dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) using [e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) generated embeddings, stored in chromadb. The frontend is served via streamlit.

## Install dependencies
The package management is done via uv. You can clone the repo and install dependencies via `uv sync` command. 

## Setup
You will need to download hte arxiv dataset (see above) and then create the embeddings. This can be done via `test_functions.py` file.


