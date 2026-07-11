"""
Pipeline stability test — DeepSeek V3 (deepseek/deepseek-chat), 10 parallel runs.
Tracks: scores, token usage, retry attempts.
Output: stability_results_v3.json
"""
import json, os, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

RESUME_PATH = os.path.join(os.path.dirname(__file__), "AGENTIC.pdf")
JD = """Senior AI/ML Engineer - Agentic Systems
Design and build multi-agent LLM systems (LangGraph, CrewAI).
Implement RAG pipelines with vector databases (ChromaDB, FAISS).
Build production FastAPI backends. Work with LoRA/QLoRA fine-tuning
and RLHF. Deploy with Docker and CI/CD. Strong Python and PyTorch.
Required: 1+ year production AI experience."""

RUNS        = 10
MODEL       = "deepseek/deepseek-chat"   # DeepSeek V3
OUTPUT_FILE = "stability_results_v3.json"
MAX_RETRIES = 3


def parse_resume() -> dict:
    from PyPDF2 import PdfReader
    from openrouter_client import OpenRouterClient, _extract_json, MODEL_FLASH
    reader = PdfReader(RESUME_PATH)
    text   = "\n".join(page.extract_text() or "" for page in reader.pages)
    llm    = OpenRouterClient()
    prompt = f"""Extract this resume into structured JSON with keys:
contact_info (name, email, phone), professional_summary,
skills (list of {{category, skills[]}}),
experience (list of {{company, position, start_date, end_date, description, achievements[], technologies[]}}),
education (list of {{institution, degree, field_of_study}}),
projects (list of {{title, description, technologies[]}}).
Return ONLY valid JSON.\n\nResume:\n{text}"""
    resp = llm.call_llm([{"role": "user", "content": prompt}], model=MODEL_FLASH, temperature=0.1)
    return _extract_json(resp)


def call_api(prompt: str, model: str, api_key: str) -> tuple[dict, dict]:
    from openrouter_client import _extract_json
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "StabilityTest",
    }
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data    = resp.json()
    usage   = data.get("usage") or {}
    msg     = data["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    return _extract_json(content), usage


def score_once(run_idx: int, resume_dict: dict, api_key: str) -> dict:
    from agentic_pipeline import SCORE_WEIGHTS

    prompt = f"""You are an expert recruiter. Evaluate this candidate against the job description.

Job Description:
{JD}

Scoring weights: skills={SCORE_WEIGHTS['skills']}, experience={SCORE_WEIGHTS['experience']}, education={SCORE_WEIGHTS['education']}, projects={SCORE_WEIGHTS['projects']}

Candidate Resume (JSON):
{json.dumps(resume_dict, indent=2)}

Return ONLY valid JSON:
{{
  "skills_score": <float 0-100>,
  "experience_score": <float 0-100>,
  "education_score": <float 0-100>,
  "projects_score": <float 0-100>,
  "strengths": ["<s1>"],
  "weaknesses": ["<w1>"],
  "matched_skills": ["<skill>"],
  "missing_skills": ["<skill>"],
  "meets_hard_requirements": <true|false>,
  "overall_recommendation": "<STRONG_MATCH|MATCH|WEAK_MATCH|NOT_QUALIFIED>",
  "reasoning": "<1-2 sentences>"
}}"""

    t0       = time.time()
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ev, usage = call_api(prompt, MODEL, api_key)
            ev["match_score"] = round(
                (ev.get("skills_score")     or 0) * SCORE_WEIGHTS["skills"]     +
                (ev.get("experience_score") or 0) * SCORE_WEIGHTS["experience"] +
                (ev.get("education_score")  or 0) * SCORE_WEIGHTS["education"]  +
                (ev.get("projects_score")   or 0) * SCORE_WEIGHTS["projects"],
                1
            )
            return {
                "run":                    run_idx,
                "model":                  MODEL,
                "match_score":            ev.get("match_score"),
                "skills_score":           ev.get("skills_score"),
                "experience_score":       ev.get("experience_score"),
                "education_score":        ev.get("education_score"),
                "projects_score":         ev.get("projects_score"),
                "overall_recommendation": ev.get("overall_recommendation"),
                "prompt_tokens":          usage.get("prompt_tokens"),
                "completion_tokens":      usage.get("completion_tokens"),
                "total_tokens":           usage.get("total_tokens"),
                "retries":                attempt - 1,
                "elapsed_s":              round(time.time() - t0, 1),
            }
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(2)

    return {"run": run_idx, "error": last_err, "retries": MAX_RETRIES - 1, "elapsed_s": round(time.time() - t0, 1)}


def fmt(v):
    return f"{v:5.1f}" if v is not None else "  N/A"

def fmti(v):
    return f"{v:6d}" if v is not None else "   N/A"


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    print(f"\n{'='*75}")
    print(f"  STABILITY TEST -- {MODEL}")
    print(f"  {RUNS} parallel runs | same resume + JD | max {MAX_RETRIES} retries")
    print(f"  Fixed weights: skills=0.40  exp=0.35  edu=0.15  proj=0.10")
    print(f"{'='*75}\n")

    print("Parsing resume (once)...")
    resume_dict = parse_resume()
    name        = (resume_dict.get("contact_info") or {}).get("name") or "Unknown"
    print(f"  Parsed: {name}\n")

    print(f"Firing {RUNS} parallel evaluations with DeepSeek V3...\n")
    t_total = time.time()

    results = [None] * RUNS
    with ThreadPoolExecutor(max_workers=RUNS) as pool:
        futures = {pool.submit(score_once, i, resume_dict, api_key): i for i in range(1, RUNS + 1)}
        for future in as_completed(futures):
            r = future.result()
            results[r["run"] - 1] = r
            if "error" in r:
                print(f"  Run {r['run']:2d} ERROR (retries={r['retries']}): {r['error']}")
            else:
                print(
                    f"  Run {r['run']:2d} -- "
                    f"match={fmt(r['match_score'])}  "
                    f"skills={fmt(r['skills_score'])}  "
                    f"exp={fmt(r['experience_score'])}  "
                    f"edu={fmt(r['education_score'])}  "
                    f"proj={fmt(r['projects_score'])}  "
                    f"tok={fmti(r['total_tokens'])}  "
                    f"retries={r['retries']}  "
                    f"{r['elapsed_s']}s"
                )

    wall = round(time.time() - t_total, 1)
    print(f"\n  All done in {wall}s wall time\n")

    good = [r for r in results if r and "error" not in r]

    print(f"{'='*75}")
    print("  RESULTS (by run number)")
    print(f"  {'Run':>3}  {'match':>6}  {'skills':>6}  {'exp':>6}  {'edu':>6}  {'proj':>6}  {'rec':<14}  {'p_tok':>6}  {'c_tok':>6}  {'retry':>5}  {'time':>5}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
    for r in results:
        if not r: continue
        if "error" in r:
            print(f"  {r['run']:>3}  ERROR: {r['error']}")
        else:
            print(
                f"  {r['run']:>3}  {fmt(r['match_score'])}  {fmt(r['skills_score'])}  "
                f"{fmt(r['experience_score'])}  {fmt(r['education_score'])}  {fmt(r['projects_score'])}  "
                f"{(r['overall_recommendation'] or '?'):<14}  "
                f"{fmti(r['prompt_tokens'])}  {fmti(r['completion_tokens'])}  "
                f"{r['retries']:>5}  {r['elapsed_s']:>4}s"
            )

    if not good:
        print("\n  No successful runs.")
        return

    print(f"\n{'='*75}")
    print("  SCORE ANALYSIS")
    print(f"  {'Field':<22}  {'Min':>6}  {'Max':>6}  {'Mean':>7}  {'Range':>7}  {'StdDev':>8}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}")
    for key in ["match_score","skills_score","experience_score","education_score","projects_score"]:
        vals = [r[key] for r in good if r.get(key) is not None]
        if not vals: continue
        mn, mx = min(vals), max(vals)
        mean   = sum(vals)/len(vals)
        std    = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
        print(f"  {key:<22}  {mn:>6.1f}  {mx:>6.1f}  {mean:>7.1f}  {mx-mn:>7.1f}  {std:>8.2f}")

    tok_vals = [r["total_tokens"] for r in good if r.get("total_tokens")]
    if tok_vals:
        print(f"\n  Token usage across {len(good)} runs:")
        p_toks = [r["prompt_tokens"] for r in good if r.get("prompt_tokens")]
        c_toks = [r["completion_tokens"] for r in good if r.get("completion_tokens")]
        print(f"    prompt tokens:     min={min(p_toks)}  max={max(p_toks)}  avg={sum(p_toks)//len(p_toks)}")
        print(f"    completion tokens: min={min(c_toks)}  max={max(c_toks)}  avg={sum(c_toks)//len(c_toks)}")
        print(f"    total tokens:      min={min(tok_vals)}  max={max(tok_vals)}  avg={sum(tok_vals)//len(tok_vals)}")

    total_retries = sum(r.get("retries", 0) for r in results if r)
    print(f"\n  Total retries across all runs: {total_retries}")

    recs = [r["overall_recommendation"] for r in good]
    print(f"\n  overall_recommendation ({len(good)} runs):")
    for rec in sorted(set(recs)):
        print(f"    {rec}: {recs.count(rec)}x")

    rng = max(r["match_score"] for r in good) - min(r["match_score"] for r in good)
    print(f"\n  Verdict: ", end="")
    if rng == 0:   print("PERFECT -- identical across all runs.")
    elif rng <= 1: print(f"EXCELLENT -- range {rng:.1f} pts.")
    elif rng <= 5: print(f"GOOD -- range {rng:.1f} pts.")
    else:          print(f"NEEDS REVIEW -- range {rng:.1f} pts.")
    print(f"{'='*75}\n")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Raw results saved to {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
