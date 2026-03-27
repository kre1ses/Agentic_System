"""
RAG knowledge base using TF-IDF retrieval.

Stores text chunks in-memory and on disk; retrieves the top-K most relevant
chunks for a given query using cosine similarity on TF-IDF vectors.
"""
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KNOWLEDGE_DIR, RAG_TOP_K


class KnowledgeBase:
    """TF-IDF based retrieval over a corpus of text chunks."""

    INDEX_PATH = KNOWLEDGE_DIR / "tfidf_index.pkl"
    CORPUS_PATH = KNOWLEDGE_DIR / "corpus.json"

    def __init__(self):
        self._chunks: list[dict] = []          # [{"text": ..., "source": ..., "tags": [...]}]
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None                    # sparse TF-IDF matrix
        self._load_or_init()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_init(self):
        if self.CORPUS_PATH.exists() and self.INDEX_PATH.exists():
            with open(self.CORPUS_PATH) as f:
                self._chunks = json.load(f)
            with open(self.INDEX_PATH, "rb") as f:
                self._vectorizer, self._matrix = pickle.load(f)
        else:
            self._seed_knowledge()
            self._rebuild_index()
            self._persist()

    def _persist(self):
        with open(self.CORPUS_PATH, "w") as f:
            json.dump(self._chunks, f, indent=2)
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump((self._vectorizer, self._matrix), f)

    def _rebuild_index(self):
        if not self._chunks:
            return
        texts = [c["text"] for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.95, min_df=1, sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, source: str = "user", tags: list[str] | None = None):
        """Add a single chunk to the knowledge base."""
        self._chunks.append({"text": text, "source": source, "tags": tags or []})
        self._rebuild_index()
        self._persist()

    def add_batch(self, items: list[dict]):
        """Add multiple chunks. Each item: {"text": ..., "source": ..., "tags": [...]}"""
        self._chunks.extend(items)
        self._rebuild_index()
        self._persist()

    def retrieve(self, query: str, top_k: int = RAG_TOP_K,
                 tag_filter: str | None = None) -> list[dict]:
        """Return the top-K most relevant chunks for a query."""
        if not self._chunks or self._vectorizer is None:
            return []

        chunks = self._chunks
        matrix = self._matrix
        if tag_filter:
            idx = [i for i, c in enumerate(self._chunks) if tag_filter in c.get("tags", [])]
            if idx:
                chunks = [self._chunks[i] for i in idx]
                matrix = self._matrix[idx]

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for i in top_idx:
            if sims[i] > 0:
                results.append({
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],
                    "tags": chunks[i].get("tags", []),
                    "score": round(float(sims[i]), 4),
                })
        return results

    def retrieve_as_context(self, query: str, top_k: int = RAG_TOP_K,
                             tag_filter: str | None = None) -> str:
        """Retrieve and format as a context string for agent prompts."""
        results = self.retrieve(query, top_k, tag_filter)
        if not results:
            return ""
        parts = ["--- Relevant knowledge ---"]
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] ({r['source']}) {r['text']}")
        parts.append("--- End of knowledge ---")
        return "\n".join(parts)

    def __len__(self):
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Seed knowledge
    # ------------------------------------------------------------------

    def _seed_knowledge(self):
        """Pre-populate the knowledge base with ML / Kaggle best practices."""
        from rag.kaggle_knowledge import KAGGLE_KNOWLEDGE_CHUNKS
        self._chunks = list(KAGGLE_KNOWLEDGE_CHUNKS)
