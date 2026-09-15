"""
build_index.py
---------------
Loads data/chunks.json, embeds every chunk, builds a FAISS index,
and persists both the index and the metadata to disk.

Run once per repo (or whenever the repo changes):
    python build_index.py
"""

import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.pkl"

# General-purpose, fast, free model. Swap for a code-aware model later:
# e.g. "jinaai/jina-embeddings-v2-base-code" for better code retrieval.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_index():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run chunker.py first: "
            f"python chunker.py /path/to/cloned/repo"
        )

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("No chunks found — check that the repo path had .py/.md files.")

    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "model_name": EMBEDDING_MODEL}, f)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    build_index()
