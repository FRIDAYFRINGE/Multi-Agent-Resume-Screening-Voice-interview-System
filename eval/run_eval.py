"""
Production evaluation harness for the screening pipeline.

Usage
-----
  # Step 1: build qrels (ground truth) for a JD
  python eval/run_eval.py build --jd "path/to/jd.txt"
  python eval/run_eval.py build --jd "Senior AI/ML Engineer..." --out eval/qrels.json

  # Step 2: evaluate the pipeline against the qrels
  python eval/run_eval.py evaluate --qrels eval/qrels.json
  python eval/run_eval.py evaluate --qrels eval/qrels.json --k 5 10 20 --out eval/results/

  # Optional: also judge any uploaded candidates not in the seed set
  python eval/run_eval.py build --jd path/to/jd.txt --judge-uploads

Outputs
-------
  eval/qrels.json           — ground truth labels (TREC-style)
  eval/results/<ts>.json    — full metric results with CIs
  Console                   — formatted comparison table + headline number
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Allow running as `python eval/run_eval.py` from the project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from project root so OPENROUTER_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

from eval.metrics import (
    ndcg_at_k, recall_at_k, ap_at_k, mrr,
    kendall_tau, spearman_r,
    jackknife_ci, jackknife_ci_corr, permutation_test,
)
from eval.qrels_builder import build_jd_entry, load_qrels, save_qrels
from eval.baselines import random_ranking, vector_only_ranking


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jd(jd_arg: str) -> str:
    """Return JD text — either the raw string or content of a file path."""
    if os.path.exists(jd_arg):
        with open(jd_arg, encoding="utf-8") as f:
            return f.read().strip()
    return jd_arg.strip()


def _get_chroma_store(chroma_path: str = "./chroma_db"):
    from agentic_pipeline import CandidateVectorStore
    return CandidateVectorStore(persist_path=chroma_path)


def _run_pipeline(jd_text: str, chroma_store) -> list[str]:
    """Run the full screening pipeline and return ordered candidate_ids."""
    from agentic_pipeline import ScreeningPipeline, ScreeningQuery
    pipeline = ScreeningPipeline(chroma_path=chroma_store.collection._client._settings.persist_directory
                                 if hasattr(chroma_store.collection, '_client') else "./chroma_db")
    query = ScreeningQuery(job_description=jd_text)
    state = pipeline.screen_candidates(query)
    rankings = state.get("final_rankings", [])
    return [r["candidate_id"] for r in rankings]


def _run_pipeline_direct(jd_text: str, chroma_path: str = "./chroma_db") -> list[str]:
    """Run the full screening pipeline using the ScreeningPipeline facade."""
    from agentic_pipeline import ScreeningPipeline, ScreeningQuery
    pipeline = ScreeningPipeline(chroma_path=chroma_path)
    query = ScreeningQuery(job_description=jd_text)
    state = pipeline.screen_candidates(query)
    rankings = state.get("final_rankings", [])
    return [r["candidate_id"] for r in rankings]


# ── Build subcommand ──────────────────────────────────────────────────────────

def cmd_build(args):
    jd_text = _load_jd(args.jd)
    out_path = args.out

    print(f"\nBuilding qrels for JD ({len(jd_text)} chars)...")

    jd_id, entry = build_jd_entry(jd_text, seed_only=True)
    print(f"  JD id:          {jd_id}")
    print(f"  Seed labels:    {entry['total_labeled']} candidates")
    for tier, count in entry["tier_counts"].items():
        print(f"    {tier}: {count}")

    if args.judge_uploads:
        print("\n  Running LLM-as-judge for uploaded candidates...")
        chroma_store = _get_chroma_store(args.chroma_path)
        from eval.llm_judge import extend_qrels
        entry = extend_qrels(entry, jd_text, chroma_store, verbose=True)

    qrels_file = load_qrels(out_path)
    qrels_file[jd_id] = entry
    save_qrels(out_path, qrels_file)

    print(f"\nSaved -> {out_path}")
    print(f"Total labeled candidates in this entry: {len(entry['qrels'])}")


# ── Evaluate subcommand ───────────────────────────────────────────────────────

def _fmt(v, w=6, d=3):
    return f"{v:{w}.{d}f}" if v is not None else " " * w + "N/A"


def _row(label, a_pt, a_lo, a_hi, b_pt, b_lo, b_hi, c_pt, c_lo, c_hi):
    def cell(pt, lo, hi):
        return f"{pt:6.3f} [{lo:5.3f},{hi:5.3f}]"
    print(f"  {label:<18} {cell(a_pt,a_lo,a_hi)}   {cell(b_pt,b_lo,b_hi)}   {cell(c_pt,c_lo,c_hi)}")


def cmd_evaluate(args):
    qrels_file = load_qrels(args.qrels)
    if not qrels_file:
        print(f"ERROR: {args.qrels} is empty or missing. Run 'build' first.")
        sys.exit(1)

    # Select JD entry
    if args.jd_id:
        jd_id = args.jd_id
    else:
        jd_id = list(qrels_file.keys())[0]
        if len(qrels_file) > 1:
            print(f"  Multiple JDs in qrels — using first: {jd_id}")

    entry = qrels_file[jd_id]
    qrels = entry["qrels"]
    jd_text = entry["jd_text"]
    k_vals = args.k

    print(f"\n{'':=<72}")
    print(f"  PIPELINE EVALUATION  |  {len(qrels)} labeled candidates  |  {jd_id}")
    print(f"{'':=<72}")
    print(f"\n  Running systems...")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print("    [1/3] Full pipeline (LangGraph + LLM reranking)...", flush=True)
    chroma_path = args.chroma_path
    pipeline_ranked = _run_pipeline_direct(jd_text, chroma_path)
    print(f"           -> returned {len(pipeline_ranked)} ranked candidates")

    # ── Vector-only baseline ──────────────────────────────────────────────────
    print("    [2/3] Vector-only baseline (ChromaDB cosine)...", flush=True)
    chroma_store = _get_chroma_store(chroma_path)
    vector_ranked = vector_only_ranking(jd_text, chroma_store)
    print(f"           -> returned {len(vector_ranked)} candidates")

    # ── Random baseline ───────────────────────────────────────────────────────
    print("    [3/3] Random baseline (seed=42)...")
    random_ranked = random_ranking(list(qrels.keys()), seed=42)

    # ── Filter ranked lists to only labeled (qrels) candidates ───────────────
    # Unlabeled candidates (uploaded resumes not in qrels) are excluded from
    # metric computation — we measure ranking quality over the labeled pool only.
    qrels_set = set(qrels.keys())

    def _filter(ranked: list[str]) -> list[str]:
        return [c for c in ranked if c in qrels_set]

    pipeline_ranked_l  = _filter(pipeline_ranked)
    vector_ranked_l    = _filter(vector_ranked)
    random_ranked_l    = _filter(random_ranked)

    unlabeled_in_pipeline = len(pipeline_ranked) - len(pipeline_ranked_l)
    if unlabeled_in_pipeline:
        print(f"  Note: {unlabeled_in_pipeline} unlabeled candidate(s) in pipeline output excluded from metrics.")

    # ── Compute metrics ───────────────────────────────────────────────────────
    results = {
        "jd_id": jd_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_labeled": len(qrels),
        "pipeline_returned": len(pipeline_ranked),
        "pipeline_labeled_returned": len(pipeline_ranked_l),
        "k_values": k_vals,
        "systems": {},
    }

    systems = {
        "pipeline":    pipeline_ranked_l,
        "vector_only": vector_ranked_l,
        "random":      random_ranked_l,
    }

    n_perm = args.n_perm

    for sname, ranked in systems.items():
        sys_results = {}
        for k in k_vals:
            sys_results[f"ndcg@{k}"]   = jackknife_ci(ndcg_at_k,  ranked, qrels, k)
            sys_results[f"recall@{k}"] = jackknife_ci(recall_at_k, ranked, qrels, k)
            sys_results[f"map@{k}"]    = jackknife_ci(ap_at_k,     ranked, qrels, k)
        sys_results["mrr"]         = jackknife_ci(lambda r, q, k_: mrr(r, q), ranked, qrels, max(k_vals))
        sys_results["kendall_tau"] = jackknife_ci_corr(kendall_tau, ranked, qrels)
        sys_results["spearman_rho"]= jackknife_ci_corr(spearman_r,  ranked, qrels)
        results["systems"][sname] = sys_results

    # Permutation test: pipeline vs random at NDCG@10 (or largest k)
    primary_k = 10 if 10 in k_vals else max(k_vals)
    p_value = permutation_test(ndcg_at_k, pipeline_ranked_l, random_ranked_l, qrels, primary_k, n_perm)
    results["permutation_test"] = {
        "metric": f"ndcg@{primary_k}",
        "sys_a": "pipeline",
        "sys_b": "random",
        "p_value": round(p_value, 4),
        "n_perm": n_perm,
    }

    # ── Print table ───────────────────────────────────────────────────────────
    p = results["systems"]["pipeline"]
    v = results["systems"]["vector_only"]
    r = results["systems"]["random"]

    header = f"  {'Metric':<18} {'Pipeline':^22}   {'Vector-Only':^22}   {'Random':^22}"
    sep = f"  {'':-<18} {'':-<22}   {'':-<22}   {'':-<22}"
    print(f"\n{header}")
    print(sep)

    def row(label, key):
        pa, pb, pc = p[key]
        va, vb, vc = v[key]
        ra, rb, rc = r[key]
        print(f"  {label:<18} {pa:6.3f} [{pb:5.3f},{pc:5.3f}]   "
              f"{va:6.3f} [{vb:5.3f},{vc:5.3f}]   "
              f"{ra:6.3f} [{rb:5.3f},{rc:5.3f}]")

    for k in k_vals:
        row(f"NDCG@{k}", f"ndcg@{k}")
    print()
    for k in k_vals:
        row(f"Recall@{k} (T1)", f"recall@{k}")
    print()
    for k in k_vals:
        row(f"MAP@{k}", f"map@{k}")
    print()
    row("MRR", "mrr")
    row("Kendall tau", "kendall_tau")
    row("Spearman rho", "spearman_rho")

    print(sep)
    pt = results["permutation_test"]
    delta = p[f"ndcg@{primary_k}"][0] - r[f"ndcg@{primary_k}"][0]
    p_str = f"p<0.001" if pt["p_value"] < 0.001 else f"p={pt['p_value']:.3f}"
    print(f"\n  Pipeline vs random: NDCG@{primary_k} delta={delta:+.3f}  "
          f"{p_str}  (permutation n={n_perm:,})")

    ndcg10_pt, ndcg10_lo, ndcg10_hi = p[f"ndcg@{primary_k}"]
    print(f"\n  Headline: NDCG@{primary_k} = {ndcg10_pt:.3f} "
          f"(95% CI {ndcg10_lo:.3f}-{ndcg10_hi:.3f}) "
          f"on {len(qrels)}-candidate labeled pool")
    print(f"\n{'':=<72}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(args.out, f"eval_{jd_id}_{ts}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved -> {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Production IR evaluation suite for the screening pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build
    p_build = sub.add_parser("build", help="Build/update qrels (ground truth) for a JD")
    p_build.add_argument("--jd", required=True, help="JD text or path to a .txt file")
    p_build.add_argument("--out", default="eval/qrels.json", help="Output qrels JSON path")
    p_build.add_argument("--judge-uploads", action="store_true",
                         help="Also LLM-judge uploaded candidates not in seed set")
    p_build.add_argument("--chroma-path", default="./chroma_db",
                         help="ChromaDB persistence path")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run pipeline + baselines and report metrics")
    p_eval.add_argument("--qrels", default="eval/qrels.json", help="Path to qrels JSON")
    p_eval.add_argument("--jd-id", default=None, help="Which JD entry to evaluate (default: first)")
    p_eval.add_argument("--k", type=int, nargs="+", default=[5, 10, 20],
                        help="Rank cutoffs (default: 5 10 20)")
    p_eval.add_argument("--out", default="eval/results", help="Directory for JSON result files")
    p_eval.add_argument("--chroma-path", default="./chroma_db",
                        help="ChromaDB persistence path")
    p_eval.add_argument("--n-perm", type=int, default=10000,
                        help="Permutation test iterations (default: 10000)")

    args = parser.parse_args()

    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
