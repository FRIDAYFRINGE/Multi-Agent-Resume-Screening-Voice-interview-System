"""
Scoring + determinism test — runs two pipeline passes and captures
match_score / ce_score / vec_sim / fused per candidate.
Writes results to rank.md (appends a new dated section).
"""
import io, sys, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from agentic_pipeline import ScreeningPipeline, ScreeningQuery

JD = """
Senior Machine Learning Engineer

We are looking for a Senior ML Engineer to join our AI platform team.
You will design and deploy large-scale machine learning systems in production.

Responsibilities:
- Build and maintain ML pipelines for model training, evaluation, and serving
- Develop and optimize deep learning models (NLP, computer vision, or recommendation)
- Collaborate with data engineers to build feature pipelines (Spark, Kafka, Airflow)
- Deploy models to production using MLflow, KubeFlow, or similar platforms
- Monitor model performance and implement A/B testing frameworks
- Mentor junior engineers and drive best practices

Requirements:
- 4+ years of experience in machine learning engineering
- Strong Python skills (PyTorch, TensorFlow, scikit-learn)
- Experience with distributed computing (Spark, Ray, Dask)
- Familiarity with cloud platforms (AWS SageMaker, GCP Vertex AI, or Azure ML)
- Experience with containerization (Docker, Kubernetes)
- Strong understanding of ML fundamentals (statistics, optimization, model evaluation)

Nice to have:
- Experience with LLMs, fine-tuning, or RLHF
- Contributions to open-source ML projects
- Experience with real-time inference systems
""".strip()

def run_once(label: str) -> list:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    pipeline = ScreeningPipeline(chroma_path="./chroma_db")
    query = ScreeningQuery(
        job_description=JD,
        required_skills=["Python", "PyTorch", "Machine Learning", "MLOps"],
        min_experience_years=4,
    )
    state = pipeline.screen_candidates(query)
    elapsed = time.time() - t0
    rankings = state.get("final_rankings", [])
    print(f"  Completed in {elapsed:.1f}s — {len(rankings)} candidates ranked")
    for r in rankings[:10]:
        print(f"  #{r.get('rank'):>2}  {r.get('candidate_id','?'):<30} "
              f"match={r.get('match_score',0):5.1f}  "
              f"ce={r.get('ce_score',0):.4f}  "
              f"vec={r.get('vector_similarity',0):.4f}  "
              f"fused={r.get('fused_score',0):.2f}")
    return rankings


def compare(r1: list, r2: list) -> dict:
    rank1 = {e["candidate_id"]: e["rank"] for e in r1}
    rank2 = {e["candidate_id"]: e["rank"] for e in r2}
    all_ids = sorted(set(rank1) | set(rank2))
    deltas = []
    for cid in all_ids:
        d = rank2.get(cid, len(r2)+1) - rank1.get(cid, len(r1)+1)
        deltas.append((cid, rank1.get(cid,"—"), rank2.get(cid,"—"), d))
    swaps = sum(1 for *_, d in deltas if d != 0)
    abs_deltas = [abs(d) for *_, d in deltas]
    mean_d = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0
    max_d  = max(abs_deltas) if abs_deltas else 0
    stable = sum(1 for d in abs_deltas if d == 0)
    return dict(deltas=deltas, swaps=swaps, mean_d=mean_d, max_d=max_d,
                stable=stable, n=len(all_ids))


def build_score_table(rankings: list) -> str:
    header = "| Rank | Candidate | match_score | ce_score | vec_sim | fused |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = ""
    for r in rankings:
        rows += (f"| {r.get('rank')} "
                 f"| {r.get('candidate_id','?')} "
                 f"| {r.get('match_score',0)} "
                 f"| {r.get('ce_score',0):.4f} "
                 f"| {r.get('vector_similarity',0):.4f} "
                 f"| {r.get('fused_score',0):.2f} |\n")
    return header + rows


def build_delta_table(deltas, model_label):
    sorted_d = sorted(deltas, key=lambda x: -abs(x[3]))
    header  = f"## {model_label}\n\n"
    n       = len(deltas)
    swaps   = sum(1 for *_, d in deltas if d != 0)
    stable  = sum(1 for *_, d in deltas if d == 0)
    mean_d  = sum(abs(d) for *_, d in deltas) / n if n else 0
    max_d   = max(abs(d) for *_, d in deltas) if deltas else 0
    header += f"**{n} candidates | {swaps} swaps | Mean |delta|: {mean_d:.1f} | Max |delta|: {max_d} | Stable: {stable}/{n}**\n\n"
    header += "| Candidate | Run 1 | Run 2 | Delta |\n|---|---|---|---|\n"
    rows = ""
    for cid, r1, r2, d in sorted_d:
        sign = f"+{d}" if d > 0 else str(d)
        rows += f"| {cid} | {r1} | {r2} | {sign} |\n"
    return header + rows


if __name__ == "__main__":
    MODEL_LABEL = "deepseek/deepseek-chat-v3-0324"

    run1 = run_once("Run 1")
    run2 = run_once("Run 2")

    cmp   = compare(run1, run2)
    print(f"\n--- Determinism Summary ---")
    print(f"Candidates : {cmp['n']}")
    print(f"Swaps      : {cmp['swaps']}")
    print(f"Mean |Δ|   : {cmp['mean_d']:.2f}")
    print(f"Max |Δ|    : {cmp['max_d']}")
    print(f"Stable     : {cmp['stable']}/{cmp['n']}")

    # Build rank.md content (full rewrite with real data)
    md = "# Ranking Determinism Report\n\n"
    md += "Two independent pipeline runs on the same JD and candidate pool.\n"
    md += "Metric: per-candidate rank shift between Run 1 and Run 2 (`Delta = Run2 - Run1`).\n"
    md += f"\nModel under test: `{MODEL_LABEL}` | temperature=0.1\n\n---\n\n"

    md += build_delta_table(cmp["deltas"], MODEL_LABEL)
    md += "\n---\n\n"

    md += f"## {MODEL_LABEL} — Full Score Breakdown (Run 1)\n\n"
    md += ("Scores from Run 1.  \n"
           "`match_score` = weighted LLM component score (0–100). "
           "`ce_score` = cross-encoder sigmoid (0–1). "
           "`vec_sim` = ChromaDB cosine similarity. "
           "`fused` = weighted RRF score ×1000 (llm 50% + ce 30% + vec 20%).\n\n")
    md += build_score_table(run1)

    md += "\n---\n\n## Summary\n\n"
    md += "| Model | Candidates | Swaps | Mean delta | Max delta | Stable |\n"
    md += "|---|---|---|---|---|---|\n"
    n, swaps, mean_d, max_d, stable = cmp["n"], cmp["swaps"], cmp["mean_d"], cmp["max_d"], cmp["stable"]
    pct = f"{stable}/{n} ({100*stable//n}%)"
    md += f"| {MODEL_LABEL} | {n} | {swaps} | {mean_d:.1f} ranks | {max_d} ranks | {pct} |\n"
    md += "\n"
    if swaps == 0:
        md += "**Verdict**: `deepseek-chat-v3-0324` produces fully deterministic rankings — 0 rank swaps across both runs.\n"
    else:
        md += f"**Verdict**: {swaps} rank swaps detected. Mean shift {mean_d:.1f} ranks, worst-case {max_d} ranks.\n"

    with open("rank.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\nrank.md updated.")
