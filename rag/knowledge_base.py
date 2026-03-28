"""
RAG knowledge base: hybrid BM25 + TF-IDF retrieval.

Retrieval strategy:
  - BM25 (rank_bm25) — keyword matching, strong on exact terms
  - TF-IDF cosine — semantic overlap, strong on synonyms / ngrams
  - Hybrid score = α * bm25_norm + (1 - α) * tfidf_norm  (α = 0.4)

If rank_bm25 is not installed, falls back to TF-IDF only.

Tag-aware retrieval:
  Each agent role has a primary tag set used to pre-filter chunks before
  scoring, so each agent gets domain-relevant knowledge without irrelevant
  context polluting the system prompt.

Experiment feedback loop:
  After each pipeline run, call kb.learn_from_experiment(chunks) to add
  structured results (best model, MSE, feature decisions) as searchable chunks.
  Future runs retrieve these to warm-start decisions.
"""
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KNOWLEDGE_DIR, RAG_TOP_K

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# Hybrid weight: fraction given to BM25 (remainder goes to TF-IDF)
_BM25_ALPHA = 0.4

# Tags each agent role should prefer when filtering chunks
_ROLE_TAG_MAP: dict[str, list[str]] = {
    "validator":   ["validation", "leakage", "regression"],
    "planner":     ["agent", "regression", "domain"],
    "explorer":    ["eda", "regression", "domain"],
    "engineer":    ["feature_engineering", "regression", "domain", "leakage"],
    "builder":     ["model_selection", "regression", "evaluation"],
    "critic":      ["evaluation", "regression", "model_selection"],
    "reporter":    ["evaluation", "regression"],
    "coordinator": ["agent", "regression"],
}


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lower tokenizer for BM25."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class KnowledgeBase:
    """Hybrid BM25 + TF-IDF knowledge base with tag-aware retrieval."""

    INDEX_PATH  = KNOWLEDGE_DIR / "tfidf_index.pkl"
    BM25_PATH   = KNOWLEDGE_DIR / "bm25_index.pkl"
    CORPUS_PATH = KNOWLEDGE_DIR / "corpus.json"

    def __init__(self):
        self._chunks: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._bm25: Any | None = None          # BM25Okapi or None
        self._load_or_init()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_init(self):
        if self.CORPUS_PATH.exists() and self.INDEX_PATH.exists():
            with open(self.CORPUS_PATH) as f:
                self._chunks = json.load(f)
            with open(self.INDEX_PATH, "rb") as f:
                self._vectorizer, self._tfidf_matrix = pickle.load(f)
            if self.BM25_PATH.exists():
                with open(self.BM25_PATH, "rb") as f:
                    self._bm25 = pickle.load(f)
        else:
            self._seed_knowledge()
            self._rebuild_index()
            self._persist()

    def _persist(self):
        with open(self.CORPUS_PATH, "w") as f:
            json.dump(self._chunks, f, indent=2)
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump((self._vectorizer, self._tfidf_matrix), f)
        if self._bm25 is not None:
            with open(self.BM25_PATH, "wb") as f:
                pickle.dump(self._bm25, f)

    def _rebuild_index(self):
        if not self._chunks:
            return
        texts = [c["text"] for c in self._chunks]

        # TF-IDF
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.95, min_df=1, sublinear_tf=True
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        # BM25
        if _BM25_AVAILABLE:
            tokenized = [_tokenize(t) for t in texts]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, source: str = "user", tags: list[str] | None = None):
        """Add a single chunk and rebuild the index."""
        self._chunks.append({"text": text, "source": source, "tags": tags or []})
        self._rebuild_index()
        self._persist()

    def add_batch(self, items: list[dict]):
        """Add multiple chunks at once (more efficient than repeated add())."""
        self._chunks.extend(items)
        self._rebuild_index()
        self._persist()

    def retrieve(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        tag_filter: str | None = None,
    ) -> list[dict]:
        """
        Hybrid BM25 + TF-IDF retrieval.

        Args:
            query:      Natural-language query string.
            top_k:      Number of results to return.
            tag_filter: If given, only chunks whose tags list contains this
                        value are considered.
        Returns:
            List of chunk dicts with an added 'score' field, sorted by score.
        """
        if not self._chunks or self._vectorizer is None:
            return []

        # ── Filter by tag ────────────────────────────────────────────
        if tag_filter:
            indices = [
                i for i, c in enumerate(self._chunks)
                if tag_filter in c.get("tags", [])
            ]
        else:
            indices = list(range(len(self._chunks)))

        if not indices:
            return []

        filtered_chunks = [self._chunks[i] for i in indices]

        # ── TF-IDF scores ────────────────────────────────────────────
        q_vec = self._vectorizer.transform([query])
        tfidf_sub = self._tfidf_matrix[indices]
        tfidf_raw = cosine_similarity(q_vec, tfidf_sub).flatten()
        tfidf_max = tfidf_raw.max() or 1.0
        tfidf_norm = tfidf_raw / tfidf_max

        # ── BM25 scores ──────────────────────────────────────────────
        if self._bm25 is not None and _BM25_AVAILABLE:
            tokenized_query = _tokenize(query)
            all_bm25 = np.array(self._bm25.get_scores(tokenized_query))
            bm25_sub = all_bm25[indices]
            bm25_max = bm25_sub.max() or 1.0
            bm25_norm = bm25_sub / bm25_max
            hybrid = _BM25_ALPHA * bm25_norm + (1 - _BM25_ALPHA) * tfidf_norm
        else:
            hybrid = tfidf_norm

        # ── Top-K ───────────────────────────────────────────────────
        top_idx = np.argsort(hybrid)[::-1][:top_k]

        results = []
        for i in top_idx:
            if hybrid[i] > 0:
                results.append({
                    "text":   filtered_chunks[i]["text"],
                    "source": filtered_chunks[i]["source"],
                    "tags":   filtered_chunks[i].get("tags", []),
                    "score":  round(float(hybrid[i]), 4),
                })
        return results

    def retrieve_for_agent(
        self,
        agent_role: str,
        query: str,
        top_k: int = RAG_TOP_K,
    ) -> list[dict]:
        """
        Retrieve chunks most relevant to a specific agent role.

        Uses the primary tag from _ROLE_TAG_MAP to filter, then scores
        within that subset. Falls back to unfiltered retrieval if too few
        results are found.
        """
        primary_tags = _ROLE_TAG_MAP.get(agent_role, [])
        results: list[dict] = []

        # Try each primary tag and merge by score (dedup by text)
        seen: set[str] = set()
        for tag in primary_tags:
            for chunk in self.retrieve(query, top_k=top_k, tag_filter=tag):
                if chunk["text"] not in seen:
                    seen.add(chunk["text"])
                    results.append(chunk)

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        # Fallback: if role has no matching tags, retrieve globally
        if not results:
            results = self.retrieve(query, top_k=top_k)

        return results

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        tag_filter: str | None = None,
        agent_role: str | None = None,
    ) -> str:
        """
        Retrieve and format as a context string ready to inject into system prompts.

        If agent_role is given, uses role-aware tag filtering.
        """
        if agent_role:
            results = self.retrieve_for_agent(agent_role, query, top_k)
        else:
            results = self.retrieve(query, top_k, tag_filter)

        if not results:
            return ""

        parts = ["--- Relevant knowledge ---"]
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] ({r['source']}) {r['text']}")
        parts.append("--- End of knowledge ---")
        return "\n".join(parts)

    def learn_from_experiment(self, chunks: list[dict]) -> None:
        """
        Ingest structured experiment results as searchable knowledge.

        Each chunk should be:
            {"text": str, "source": "experiment", "tags": [...]}

        Call this at the end of each pipeline run so future runs can
        retrieve which models and features worked best.
        """
        if not chunks:
            return
        # Avoid duplicating identical texts
        existing_texts = {c["text"] for c in self._chunks}
        new_chunks = [c for c in chunks if c["text"] not in existing_texts]
        if new_chunks:
            self.add_batch(new_chunks)

    def __len__(self):
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Seed knowledge
    # ------------------------------------------------------------------

    def _seed_knowledge(self):
        from rag.kaggle_knowledge import KAGGLE_KNOWLEDGE_CHUNKS
        self._chunks = list(KAGGLE_KNOWLEDGE_CHUNKS)
