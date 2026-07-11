"""
Cross-encoder reranker — cross-encoder/ms-marco-MiniLM-L-6-v2.
Local, deterministic, free. Scores (query, document) pairs jointly,
far more accurate than bi-encoder cosine for relevance ranking.
"""
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _model


def ce_scores(jd_text: str, docs: list) -> list:
    """Sigmoid-normalised relevance scores (0-1) for (JD, doc) pairs."""
    import numpy as np
    if not docs:
        return []
    logits = _get_model().predict([(jd_text, d or "") for d in docs])
    return [float(s) for s in (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))))]
