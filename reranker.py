"""
Cross-encoder reranker — cross-encoder/ms-marco-MiniLM-L-6-v2.
Local, deterministic, free. Scores (query, document) pairs jointly,
far more accurate than bi-encoder cosine for relevance ranking.

The model is a process-wide singleton. Call preload() once at app startup so
the first screening run doesn't pay the construction cost.
"""
import os
import threading

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_lock = threading.Lock()


def _set_offline(flag: bool) -> None:
    """Toggle HuggingFace Hub offline mode.

    The env var alone is not enough: huggingface_hub snapshots it into
    constants.HF_HUB_OFFLINE at import time, so if anything imported the
    library before us (chromadb does), mutating os.environ is a no-op.
    Patch both and the setting holds either way.
    """
    os.environ["HF_HUB_OFFLINE"] = "1" if flag else "0"
    try:
        from huggingface_hub import constants
        constants.HF_HUB_OFFLINE = flag
    except Exception:
        pass  # library absent or restructured — env var still applies


def _build():
    """Construct the CrossEncoder, preferring the local cache.

    A cold CrossEncoder() spends ~49s round-tripping the Hub to revision-check
    weights it already has on disk; offline mode skips that entirely (~0.5s).
    Try offline first, then fall back to a networked load so a fresh machine
    with an empty cache can still download the model on first run.
    """
    from sentence_transformers import CrossEncoder
    prev = os.environ.get("HF_HUB_OFFLINE")
    _set_offline(True)
    try:
        try:
            return CrossEncoder(MODEL_NAME, max_length=512)
        except Exception:
            _set_offline(False)  # nothing cached → allow the download
            return CrossEncoder(MODEL_NAME, max_length=512)
    finally:
        # Leave global state exactly as we found it; other libraries in this
        # process may rely on the Hub being reachable.
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                from huggingface_hub import constants
                constants.HF_HUB_OFFLINE = False
            except Exception:
                pass
        else:
            _set_offline(prev == "1")


def _get_model():
    global _model
    if _model is None:
        with _lock:  # concurrent first-callers must not each build a model
            if _model is None:
                _model = _build()
    return _model


def preload() -> bool:
    """Warm the singleton ahead of first use. Returns True if the model is ready.

    Safe to call more than once and safe to call off the main thread; failure
    is non-fatal because ce_scores() degrades gracefully without a model.
    """
    try:
        _get_model()
        return True
    except Exception:
        return False


def ce_scores(jd_text: str, docs: list) -> list:
    """Sigmoid-normalised relevance scores (0-1) for (JD, doc) pairs."""
    import numpy as np
    if not docs:
        return []
    logits = _get_model().predict([(jd_text, d or "") for d in docs])
    return [float(s) for s in (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))))]
