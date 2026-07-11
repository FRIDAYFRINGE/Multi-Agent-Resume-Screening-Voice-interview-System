"""
Baseline ranking systems for comparison against the full pipeline.

  random_ranking    — deterministic shuffle (fixed seed for reproducibility)
  vector_only_ranking — ChromaDB cosine similarity, no LLM reranking
"""
import random as _random
from typing import Optional


def random_ranking(candidate_ids: list[str], seed: int = 42) -> list[str]:
    """Return a deterministically shuffled list of candidate IDs.
    Fixed seed ensures the same 'random' baseline across all evaluation runs."""
    shuffled = list(candidate_ids)
    rng = _random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def vector_only_ranking(
    jd_text: str,
    chroma_store,
    top_k: Optional[int] = None,
) -> list[str]:
    """Rank all candidates by cosine similarity to the JD embedding.
    No LLM call — pure vector-space ranking.

    Returns candidate IDs sorted from most to least similar.
    If top_k is None, retrieves all candidates in the collection.
    """
    n = chroma_store.count()
    if n == 0:
        return []
    k = top_k if top_k is not None else n

    results = chroma_store.collection.query(
        query_texts=[jd_text],
        n_results=min(k, n),
        include=["distances"],
    )
    if not results or not results["ids"] or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    distances = results["distances"][0]  # lower = more similar for cosine

    # Sort ascending by distance (already sorted by ChromaDB, but make explicit)
    paired = sorted(zip(distances, ids), key=lambda x: x[0])
    return [cid for _, cid in paired]
