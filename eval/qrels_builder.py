"""
Ground truth (qrels) management in TREC-style JSON format.

Qrels file schema:
{
  "jd_<16-char-hash>": {
    "jd_text": "...",
    "created_at": "ISO timestamp",
    "qrels": {
      "candidate_id": relevance_grade,   # 0-4
      ...
    }
  }
}

Relevance grades:
  4 — Tier 1: strongly relevant (agentic/RAG/LLM stack)
  3 — Tier 2: relevant (solid ML, partial stack)
  2 — Tier 3: marginally relevant (general software / ML adjacent)
  1 — Tier 4: not relevant (wrong domain)
  0 — Unjudged or explicitly not relevant
"""
import json
import os
import sys
from datetime import datetime, timezone

TIER_TO_GRADE: dict[int, int] = {1: 4, 2: 3, 3: 2, 4: 1}


def _parent_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_seed_qrels() -> dict[str, int]:
    """Extract tier labels from seed_candidates.SEEDS.
    Returns {candidate_id: relevance_grade} for all 50 seeded candidates."""
    parent = _parent_dir()
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from seed_candidates import SEEDS  # noqa: PLC0415
    return {seed[0]: TIER_TO_GRADE[seed[2]] for seed in SEEDS}


def load_qrels(path: str) -> dict:
    """Load full qrels file (dict keyed by jd_id)."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_qrels(path: str, qrels_file: dict) -> None:
    """Persist qrels file to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qrels_file, f, indent=2, ensure_ascii=False)


def build_jd_entry(jd_text: str, seed_only: bool = True) -> tuple[str, dict]:
    """Build a single JD entry for the qrels file.

    Returns (jd_id, entry_dict) where jd_id is the 16-char hash key.
    seed_only=True uses only the SEEDS-derived labels (no LLM judge).
    """
    parent = _parent_dir()
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from database import make_jd_hash  # noqa: PLC0415

    jd_id = f"jd_{make_jd_hash(jd_text)}"
    qrels = build_seed_qrels()

    entry = {
        "jd_text": jd_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qrels": qrels,
        "label_source": "seed_candidates.SEEDS",
        "tier_counts": {
            f"tier_{t}": sum(1 for g in qrels.values() if g == grade)
            for t, grade in TIER_TO_GRADE.items()
        },
        "total_labeled": len(qrels),
    }
    return jd_id, entry
