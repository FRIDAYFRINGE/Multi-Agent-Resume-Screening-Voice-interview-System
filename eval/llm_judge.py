"""
LLM-as-judge for assigning relevance grades to uploaded (non-seed) candidates.

Uses DeepSeek V3 with 3-run self-consistency (majority vote).
Only needed for candidates absent from seed_candidates.SEEDS.
"""
import json
import os
import sys
from collections import Counter
from typing import Optional

_RUBRIC = """
You are an expert recruiter evaluating a candidate for a role.
Assign a relevance grade using this scale:

  Grade 4 — Strongly relevant
    Has direct experience with: agentic systems (LangGraph, CrewAI, AutoGPT patterns),
    RAG pipelines (ChromaDB, FAISS, Weaviate, hybrid search), AND LLM deployment or
    fine-tuning (LoRA, vLLM, OpenAI API, prompt engineering). Tier-1 candidate.

  Grade 3 — Relevant
    Solid ML/NLP background with RAG or LLM API experience.
    Missing one of the three agentic/RAG/LLM pillars. Would need 2-3 months ramp.

  Grade 2 — Marginally relevant
    General software engineer with some ML exposure (sklearn, pandas, basic models).
    No production LLM or RAG experience. Would need 6+ months ramp.

  Grade 1 — Not relevant
    No ML background or wrong domain entirely (pure frontend, Android, networking, etc.).

Return ONLY valid JSON:
{"grade": <integer 1-4>, "reason": "<one sentence>"}
"""


def _parent_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def judge_relevance(
    jd_text: str,
    resume_dict: dict,
    n_runs: int = 3,
) -> tuple[int, float]:
    """Judge relevance of a single candidate to a JD.

    Returns (majority_grade, agreement_rate).
    agreement_rate: fraction of runs matching the majority (0.67 for 2/3, 1.0 for 3/3).
    Returns grade 0 if all runs disagree (unjudged — excluded from metrics).
    """
    parent = _parent_dir()
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from openrouter_client import OpenRouterClient, MODEL_V3, _extract_json  # noqa: PLC0415

    client = OpenRouterClient()
    prompt = f"""{_RUBRIC}

Job Description:
{jd_text[:2000]}

Candidate Resume (JSON):
{json.dumps(resume_dict, indent=2)[:3000]}
"""
    grades = []
    for _ in range(n_runs):
        try:
            resp = client.call_llm(
                [{"role": "user", "content": prompt}],
                model=MODEL_V3,
                temperature=0.1,
                max_tokens=128,
            )
            parsed = _extract_json(resp)
            grade = int(parsed.get("grade", 0))
            if 1 <= grade <= 4:
                grades.append(grade)
        except Exception:
            continue

    if not grades:
        return 0, 0.0

    counter = Counter(grades)
    majority_grade, majority_count = counter.most_common(1)[0]
    agreement = majority_count / n_runs

    # All three disagree → unjudged
    if majority_count == 1 and n_runs == 3:
        return 0, 0.0

    return majority_grade, round(agreement, 2)


def extend_qrels(
    qrels_entry: dict,
    jd_text: str,
    chroma_store,
    verbose: bool = True,
) -> dict:
    """Add LLM-judged grades for any ChromaDB candidate not already in qrels_entry['qrels'].

    Modifies and returns the qrels_entry in place.
    """
    existing_ids = set(qrels_entry.get("qrels", {}).keys())

    # Fetch all ChromaDB IDs
    try:
        all_results = chroma_store.collection.get(include=["metadatas"])
        all_ids = all_results.get("ids", [])
        all_metas = all_results.get("metadatas", [])
    except Exception as e:
        print(f"  [llm_judge] Could not fetch ChromaDB candidates: {e}")
        return qrels_entry

    to_judge = [(cid, meta) for cid, meta in zip(all_ids, all_metas) if cid not in existing_ids]

    if not to_judge:
        if verbose:
            print("  [llm_judge] No new candidates to judge.")
        return qrels_entry

    if verbose:
        print(f"  [llm_judge] Judging {len(to_judge)} uploaded candidate(s) (3 runs each)...")

    qrels_entry.setdefault("qrels", {})
    qrels_entry.setdefault("judge_metadata", {})

    for cid, meta in to_judge:
        try:
            resume_dict = json.loads(meta.get("resume_json", "{}"))
        except Exception:
            resume_dict = {}

        grade, agreement = judge_relevance(jd_text, resume_dict)
        qrels_entry["qrels"][cid] = grade
        qrels_entry["judge_metadata"][cid] = {
            "grade": grade,
            "agreement": agreement,
            "source": "llm_judge_v3",
        }
        status = f"grade={grade} agree={agreement:.0%}" if grade > 0 else "unjudged"
        if verbose:
            print(f"    {cid}: {status}")

    return qrels_entry
