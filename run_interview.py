"""
CLI entry point — run a live voice interview for a screened candidate.

Usage:
    python run_interview.py                        # uses AGENTIC.pdf + default JD
    python run_interview.py --resume AGENTIC.pdf  # specify resume
    python run_interview.py --whisper small        # better STT accuracy
"""
import sys
import json
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


JD = """
Senior AI/ML Engineer - Agentic Systems
Design and build multi-agent LLM systems (LangGraph, CrewAI).
Implement RAG pipelines with vector databases (ChromaDB, FAISS).
Build production FastAPI backends. Work with LoRA/QLoRA fine-tuning
and RLHF. Deploy with Docker and CI/CD. Strong Python and PyTorch.
Required: 1+ year production AI experience.
"""


def main():
    parser = argparse.ArgumentParser(description="Live voice interview simulator")
    parser.add_argument("--resume",   default="AGENTIC.pdf",  help="Path to resume PDF")
    parser.add_argument("--whisper",  default="base",          help="Whisper model size: tiny|base|small|medium")
    parser.add_argument("--followups", type=int, default=1,   help="Max follow-up questions per answer")
    parser.add_argument("--jd",       default=None,           help="Path to a .txt file with custom JD")
    args = parser.parse_args()

    jd = JD
    if args.jd and os.path.exists(args.jd):
        with open(args.jd, encoding="utf-8") as f:
            jd = f.read()

    # ── Step 1: Parse resume + run screening pipeline ─────────────────────────
    print("[SETUP] Parsing resume and running screening pipeline...")
    from resume_parser import ResumeParser
    from openrouter_client import OpenRouterClient
    from agentic_pipeline import ScreeningPipeline, ScreeningQuery

    llm = OpenRouterClient()
    pipeline = ScreeningPipeline()

    resume = ResumeParser().parse_resume(args.resume, llm)
    candidate_id = resume.contact_info.name.lower().replace(" ", "_") if resume.contact_info and resume.contact_info.name else "candidate"
    pipeline.add_candidate(resume, candidate_id)

    state = pipeline.screen_candidates(ScreeningQuery(
        job_description=jd,
        required_skills=["LangGraph", "ChromaDB", "FastAPI", "RAG", "PyTorch"],
        min_experience_years=1,
    ))

    # ── Step 2: Find this candidate's interview questions ─────────────────────
    iq = state["interview_questions"].get(candidate_id)

    if not iq:
        # candidate may not have been recommended — show ranking and exit
        print("\n[INFO] Candidate was not recommended for interview by the screening pipeline.")
        for r in state["final_rankings"]:
            print(f"  #{r['rank']} {r['candidate_name']} — {r['match_score']}/100 — {r['overall_recommendation']}")
        sys.exit(0)

    candidate_name = resume.contact_info.name or candidate_id
    print(f"\n[SETUP] Candidate: {candidate_name}")
    print(f"[SETUP] Match score: {iq.get('match_score')}/100  |  Rank: #{iq.get('rank')}")
    print(f"[SETUP] Questions ready: {sum(len(v) for v in iq.values() if isinstance(v, list))}")
    print()

    # ── Step 3: Run the live interview ────────────────────────────────────────
    from interview_simulator import InterviewSimulator

    simulator = InterviewSimulator(
        whisper_model=args.whisper,
        max_follow_ups=args.followups
    )

    input("[SETUP] Press Enter to start the interview (make sure your mic is connected)...")

    report = simulator.run(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        job_description=jd,
        interview_questions=iq
    )

    # ── Step 4: Save report ───────────────────────────────────────────────────
    report_path = f"interview_report_{candidate_id}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
