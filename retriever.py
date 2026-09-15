"""
retriever.py
------------
Loads the persisted FAISS index + metadata (no re-embedding of the repo)
and exposes a simple search() function.
"""

import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.pkl"


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.model = SentenceTransformer(meta["model_name"])

    def search(self, query: str, top_k: int = 5):
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results
