"""
Interview UI — FastAPI backend + HTML/JS frontend
Real-time voice interview: TTS via edge-tts, STT via faster-whisper
"""
import asyncio
import base64
import io
import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

from ui import HTML  # noqa: E402  (imported after dotenv)
from database import (save_candidate, save_job, save_interview_report,
                      list_jobs, get_job, list_reports, is_connected,
                      resolve_candidate_id, add_resume_version,
                      find_resume_by_filename, update_resume_version_inplace,
                      make_jd_hash, find_job_by_hash)

# ── STT provider ───────────────────────────────────────────────────────────────
# "local" → faster-whisper large-v3  (free, no API key, slow on CPU)
# "groq"  → Groq cloud Whisper large-v3  (free tier, fast, needs GROQ_API_KEY in .env)
STT_PROVIDER = "groq"

app = FastAPI(title="AI Interview Assistant")

# ── In-memory session store ────────────────────────────────────────────────────
sessions: dict = {}  # session_id → interview state

# Thresholds for auto-interviewing uploaded candidates
_STRONG_FIT_SCORE = 70
_GOOD_FIT_SCORE   = 50


def _match_level(match_score: float, overall_recommendation: str) -> str:
    rec = overall_recommendation or ""
    if match_score >= _STRONG_FIT_SCORE or rec == "STRONG_MATCH":
        return "Strong Fit"
    if match_score >= _GOOD_FIT_SCORE or rec == "MATCH":
        return "Good Fit"
    if match_score >= 30 or rec == "WEAK_MATCH":
        return "Potential Fit"
    return "Not Recommended"


def _generate_iq(llm, jd: str, ranking: dict, resume_dict: dict) -> dict:
    """Generate interview questions for one candidate (blocking LLM call)."""
    from interview_simulator import _parse_json
    prompt = f"""You are a senior technical interviewer. Generate a tailored interview question set for this candidate.

Role being hired for:
{jd}

Candidate resume:
{json.dumps(resume_dict, indent=2)}

Screening evaluation:
- Match score: {ranking.get('match_score')}/100
- Strengths: {json.dumps(ranking.get('strengths', []))}
- Weaknesses / gaps: {json.dumps(ranking.get('weaknesses', []))}
- Missing skills: {json.dumps(ranking.get('missing_skills', []))}
- Matched skills: {json.dumps(ranking.get('matched_skills', []))}

Return ONLY valid JSON:
{{
  "candidate_name": "<name>",
  "technical_depth": [
    {{"question": "<q>", "what_to_look_for": "<signal>", "follow_up": "<follow up if shallow>"}}
  ],
  "gap_probing": [
    {{"gap": "<missing skill>", "question": "<q>", "what_to_look_for": "<signal>"}}
  ],
  "project_specific": [
    {{"project": "<project name>", "question": "<q>", "what_to_look_for": "<signal>"}}
  ],
  "system_design": [
    {{"question": "<q>", "what_to_look_for": "<signal>"}}
  ],
  "behavioral": [
    {{"question": "<q>", "what_to_look_for": "<signal>"}}
  ]
}}"""
    resp = llm.call_llm([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=1400)
    return _parse_json(resp)


def _resume_similarity(a: dict, b: dict) -> float:
    """Jaccard similarity on word sets of two resume dicts (0.0 – 1.0)."""
    import re as _re
    def words(d):
        return set(_re.findall(r'\w+', json.dumps(d).lower()))
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _evaluate_candidate(llm, jd: str, cid: str, resume_dict: dict) -> dict:
    """Score one candidate against a JD (used in the fast-path re-run)."""
    from interview_simulator import _parse_json
    name = (resume_dict.get("contact_info") or {}).get("name") or cid
    prompt = f"""You are an expert recruiter. Evaluate this candidate against the job description.

Job Description:
{jd}

Candidate Resume (JSON):
{json.dumps(resume_dict, indent=2)}

Return ONLY valid JSON:
{{
  "match_score": <float 0-100>,
  "skills_score": <float 0-100>,
  "experience_score": <float 0-100>,
  "education_score": <float 0-100>,
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<gap1>", "<gap2>"],
  "matched_skills": ["<skill1>"],
  "missing_skills": ["<skill1>"],
  "meets_hard_requirements": <true|false>,
  "overall_recommendation": "<STRONG_MATCH|MATCH|WEAK_MATCH|NOT_QUALIFIED>",
  "reasoning": "<1-2 sentence summary>"
}}"""
    try:
        resp = llm.call_llm([{"role": "user", "content": prompt}], temperature=0.2)
        ev = _parse_json(resp)
    except Exception as exc:
        ev = {"match_score": 0, "overall_recommendation": "ERROR", "reasoning": str(exc)}
    ev["candidate_id"] = cid
    ev["candidate_name"] = name
    return ev


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.get("/api/pool-count")
async def pool_count():
    from agentic_pipeline import CandidateVectorStore
    store = CandidateVectorStore(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    return {"count": store.count(), "mongo": is_connected()}


@app.get("/api/jobs")
async def get_jobs():
    """List past screening runs from MongoDB."""
    try:
        jobs = list_jobs()
        for j in jobs:
            j["_id"] = str(j["_id"])
            if j.get("run_at"):
                j["run_at"] = j["run_at"].isoformat()
        return {"jobs": jobs}
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/jobs/{job_id}")
async def get_job_detail(job_id: str):
    """Full screening result for one job run."""
    try:
        job = get_job(job_id)
        if not job:
            return JSONResponse({"error": "Job not found"})
        job["_id"] = str(job["_id"])
        if job.get("run_at"):
            job["run_at"] = job["run_at"].isoformat()
        return job
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/reports")
async def get_reports():
    """List completed interview reports."""
    try:
        reports = list_reports()
        for r in reports:
            if r.get("completed_at"):
                r["completed_at"] = r["completed_at"].isoformat()
        return {"reports": reports}
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/api/screen")
async def screen_endpoint(
    resumes: List[UploadFile] = File(default=[]),
    jd: str = Form(""),
    skills: str = Form(""),
    max_follow_ups: int = Form(1)
):
    from agentic_pipeline import ScreeningPipeline, ScreeningQuery
    from interview_simulator import _flatten_questions

    try:
        pipeline = ScreeningPipeline()

        # Parse and add any uploaded resumes to the pool
        uploaded_cids = []
        uploaded = [r for r in resumes if r and r.filename]
        if uploaded:
            from resume_parser import ResumeParser
            from openrouter_client import OpenRouterClient
            parser = ResumeParser()
            llm = OpenRouterClient()
            for upload in uploaded:
                suffix = os.path.splitext(upload.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await upload.read())
                    tmp_path = tmp.name
                try:
                    parsed = parser.parse_resume(tmp_path, llm)
                    ci     = parsed.contact_info
                    email  = (ci.email or "").strip().lower() if ci else ""
                    name   = (ci.name or "candidate") if ci else "candidate"

                    # Resolve IDs (reads MongoDB latest_sub_id — does not write)
                    base_id, chroma_id, sub_id, doc_key = resolve_candidate_id(
                        email, name, pipeline.store
                    )

                    # ── Filename-based deduplication ──────────────────────────
                    existing_entry = (
                        find_resume_by_filename(doc_key, upload.filename)
                        if is_connected() and upload.filename else None
                    )

                    if existing_entry:
                        ex_chroma_id = existing_entry["chroma_id"]
                        ex_resume    = pipeline.store.get_candidate_data(ex_chroma_id) or {}
                        similarity   = _resume_similarity(parsed.dict(), ex_resume)

                        if similarity >= 0.90:
                            # Same resume re-uploaded — skip, reuse existing entry
                            uploaded_cids.append(ex_chroma_id)
                            continue

                        else:
                            # Same filename, meaningfully different content — overwrite
                            pipeline.add_candidate(parsed, ex_chroma_id)
                            uploaded_cids.append(ex_chroma_id)
                            if is_connected():
                                update_resume_version_inplace(
                                    doc_key, ex_chroma_id, parsed.dict()
                                )
                            continue

                    # ── New filename → new sub_id ─────────────────────────────
                    pipeline.add_candidate(parsed, chroma_id)
                    uploaded_cids.append(chroma_id)
                    if is_connected():
                        add_resume_version(
                            doc_key, base_id, email, name,
                            chroma_id, sub_id,
                            upload.filename or "", parsed.dict(),
                            source="uploaded",
                        )
                finally:
                    os.unlink(tmp_path)

        if pipeline.store.count() == 0:
            return JSONResponse({"error": "No candidates in pool. Upload a resume or run seed_candidates.py first."})

        required_skills = [s.strip() for s in skills.split(",") if s.strip()]

        # ── JD deduplication check ─────────────────────────────────────────────
        h            = make_jd_hash(jd)
        existing_job = find_job_by_hash(h) if is_connected() else None

        # ── Fast path: same JD seen before + new resumes uploaded ─────────────
        if existing_job and uploaded_cids:
            from openrouter_client import OpenRouterClient as _ORC
            _llm = _ORC()

            # Evaluate only the newly uploaded candidates
            uploaded_evals: list = []
            def _eval_upload(cid: str):
                rd = pipeline.store.get_candidate_data(cid)
                if rd:
                    uploaded_evals.append(_evaluate_candidate(_llm, jd, cid, rd))

            with ThreadPoolExecutor(max_workers=min(5, len(uploaded_cids))) as _pool:
                list(_pool.map(_eval_upload, uploaded_cids))

            # Merge: new evals on top of cached pool rankings
            cached_rankings = [
                r for r in (existing_job.get("rankings") or [])
                if r.get("candidate_id") not in set(uploaded_cids)  # avoid dupe if re-uploaded
            ]
            merged = uploaded_evals + cached_rankings
            merged.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            for i, r in enumerate(merged):
                r["rank"] = i + 1
                r["match_level"] = _match_level(
                    r.get("match_score", 0), r.get("overall_recommendation", "")
                )
                if r.get("candidate_id") in set(uploaded_cids):
                    r["newly_uploaded"] = True

            # Create sessions for qualifying candidates:
            #   - all uploaded Strong/Good Fit (no cap)
            #   - top 5 pool Strong/Good Fit (from cached rankings)
            sessions_out: dict = {}
            uploaded_fits = [
                r for r in merged
                if r.get("newly_uploaded")
                and r.get("match_level") in ("Strong Fit", "Good Fit")
            ]
            pool_fits = [
                r for r in merged
                if not r.get("newly_uploaded")
                and r.get("match_level") in ("Strong Fit", "Good Fit")
            ][:5]
            needs_iq = uploaded_fits + pool_fits

            def _make_fast_session(ranking: dict):
                cid = ranking["candidate_id"]
                rd  = pipeline.store.get_candidate_data(cid)
                if not rd:
                    return
                try:
                    iq = _generate_iq(_llm, jd, ranking, rd)
                except Exception:
                    return
                iq["candidate_id"] = cid
                iq["rank"]         = ranking.get("rank")
                iq["match_score"]  = ranking.get("match_score")
                sid = str(uuid.uuid4())
                question_queue = _flatten_questions(iq)
                sessions[sid] = {
                    "candidate_id":   cid,
                    "candidate_name": ranking.get("candidate_name", cid),
                    "job_description": jd,
                    "job_id":         str(existing_job["_id"]),
                    "question_queue": question_queue,
                    "current_idx":    0,
                    "follow_up_count": 0,
                    "max_follow_ups": max_follow_ups,
                    "transcript":     [],
                    "answer_scores":  [],
                    "match_score":    iq.get("match_score"),
                    "iq":             iq,
                }
                sessions_out[cid] = {"session_id": sid, "total_questions": len(question_queue)}

            if needs_iq:
                with ThreadPoolExecutor(max_workers=min(5, len(needs_iq))) as _pool:
                    list(_pool.map(_make_fast_session, needs_iq))

            current_pool = pipeline.store.count()
            old_pool     = existing_job.get("pool_size", 0)
            reused_at    = existing_job.get("run_at")

            return {
                "rankings":        merged,
                "sessions":        sessions_out,
                "pool_size":       current_pool,
                "uploaded_cids":   uploaded_cids,
                "reused_job_id":   str(existing_job["_id"]),
                "reused_job_date": reused_at.isoformat() if reused_at else "",
                "pool_changed":    current_pool != old_pool,
                "old_pool_size":   old_pool,
            }

        # ── Full pipeline path ─────────────────────────────────────────────────
        state = pipeline.screen_candidates(ScreeningQuery(
            job_description=jd,
            required_skills=required_skills,
            min_experience_years=1,
        ))

        # Persist screening run to MongoDB (include JD hash for future dedup)
        job_id = None
        if is_connected():
            try:
                job_id = save_job({
                    "jd_hash":         h,
                    "job_description": jd,
                    "required_skills": required_skills,
                    "pool_size":       pipeline.store.count(),
                    "rankings":        state["final_rankings"],
                    "critique":        state.get("critique"),
                })
            except Exception:
                pass

        # Attach match_level label to every ranking entry
        for r in state["final_rankings"]:
            r["match_level"] = _match_level(
                r.get("match_score", 0),
                r.get("overall_recommendation", ""),
            )

        # Create sessions for every candidate that received interview questions
        sessions_out: dict = {}
        for cid, iq in state["interview_questions"].items():
            sid = str(uuid.uuid4())
            question_queue = _flatten_questions(iq)
            ranking = next((r for r in state["final_rankings"] if r["candidate_id"] == cid), {})
            sessions[sid] = {
                "candidate_id":    cid,
                "candidate_name":  ranking.get("candidate_name", cid),
                "job_description": jd,
                "job_id":          job_id,
                "question_queue":  question_queue,
                "current_idx":     0,
                "follow_up_count": 0,
                "max_follow_ups":  max_follow_ups,
                "transcript":      [],
                "answer_scores":   [],
                "match_score":     iq.get("match_score"),
                "iq":              iq,
            }
            sessions_out[cid] = {"session_id": sid, "total_questions": len(question_queue)}

        # For uploaded candidates that are Strong/Good Fit but didn't make the
        # top-5 recommended cut, generate interview questions and create sessions.
        needs_iq = [
            r for r in state["final_rankings"]
            if r.get("candidate_id") in uploaded_cids
            and r.get("candidate_id") not in sessions_out
            and r.get("match_level") in ("Strong Fit", "Good Fit")
        ]

        if needs_iq:
            from openrouter_client import OpenRouterClient as _ORC
            _llm = _ORC()

            def _make_session(ranking: dict):
                cid = ranking["candidate_id"]
                resume_dict = pipeline.store.get_candidate_data(cid)
                if not resume_dict:
                    return
                try:
                    iq = _generate_iq(_llm, jd, ranking, resume_dict)
                except Exception:
                    return
                iq["candidate_id"] = cid
                iq["rank"]         = ranking.get("rank")
                iq["match_score"]  = ranking.get("match_score")
                sid = str(uuid.uuid4())
                question_queue = _flatten_questions(iq)
                sessions[sid] = {
                    "candidate_id":    cid,
                    "candidate_name":  ranking.get("candidate_name", cid),
                    "job_description": jd,
                    "job_id":          job_id,
                    "question_queue":  question_queue,
                    "current_idx":     0,
                    "follow_up_count": 0,
                    "max_follow_ups":  max_follow_ups,
                    "transcript":      [],
                    "answer_scores":   [],
                    "match_score":     iq.get("match_score"),
                    "iq":              iq,
                }
                sessions_out[cid] = {"session_id": sid, "total_questions": len(question_queue)}

            with ThreadPoolExecutor(max_workers=min(5, len(needs_iq))) as _pool:
                list(_pool.map(_make_session, needs_iq))

        # Tag newly uploaded candidates so the UI can highlight them
        for r in state["final_rankings"]:
            if r.get("candidate_id") in uploaded_cids:
                r["newly_uploaded"] = True

        return {
            "rankings":      state["final_rankings"],
            "sessions":      sessions_out,
            "pool_size":     pipeline.store.count(),
            "uploaded_cids": uploaded_cids,
        }

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "detail": traceback.format_exc()})


@app.post("/api/submit-answer")
async def submit_answer(
    audio: UploadFile = File(...),
    session_id: str = Form(""),
    question_idx: int = Form(0)
):
    """Transcribe audio + evaluate answer, push results via WS"""
    if session_id not in sessions:
        return JSONResponse({"error": "Session not found"})

    sess = sessions[session_id]
    ws_conn = sess.get("ws")

    try:
        # Save uploaded audio (WAV from browser conversion)
        suffix = os.path.splitext(audio.filename or "answer.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            audio_path = tmp.name

        # Build domain vocabulary prompt (used by both providers)
        q_text = sess["question_queue"][sess["current_idx"]].get("question", "")
        TECH_PROMPT = (
            "JD parsing, job description, routing logic, deterministic routing, "
            "schema validation, field extraction, fallback strategy, rule-based parser, "
            "LangGraph, ChromaDB, FAISS, BM25, RAG pipeline, semantic search, "
            "dense embeddings, Reciprocal Rank Fusion, RRF, reranking, chunking, "
            "PyTorch, FastAPI, LoRA, QLoRA, RLHF, fine-tuning, vector database, "
            "cosine similarity, LLM, transformer, hyperparameter, agentic pipeline, "
            f"end-to-end latency, throughput, Docker, CI/CD. Question: {q_text}"
        )

        # Transcribe — provider selected by STT_PROVIDER
        if STT_PROVIDER == "groq":
            from voice_io import GroqSTT
            stt = sess.get("stt")
            if not stt:
                stt = GroqSTT()
                sess["stt"] = stt
            text = stt.transcribe_file(audio_path, prompt=TECH_PROMPT)
            os.unlink(audio_path)
        else:
            from voice_io import SpeechToText
            stt = sess.get("stt")
            if not stt:
                stt = SpeechToText(model_size="large-v3")
                sess["stt"] = stt
            segments, _ = stt.model.transcribe(
                audio_path, language="en", beam_size=5,
                initial_prompt=TECH_PROMPT, vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            os.unlink(audio_path)

        # Push raw transcription — evaluator LLM handles STT noise during scoring
        if ws_conn:
            await ws_conn.send_json({"type": "transcription", "text": text})

        # Guard: no real speech detected — skip LLM, return zero score
        if not text or len(text.split()) < 4:
            empty_eval = {
                "score": 0, "depth": "shallow",
                "hits": [], "misses": ["No answer given"],
                "needs_follow_up": False, "follow_up_reason": "",
                "brief_feedback": "No speech detected or answer too short.",
            }
            sess["transcript"].append({
                "question_idx": sess["current_idx"],
                "category": sess["question_queue"][sess["current_idx"]]["category"],
                "question": sess["question_queue"][sess["current_idx"]]["question"],
                "answer": text, "evaluation": empty_eval,
            })
            sess["answer_scores"].append({"score": 0, "category": sess["question_queue"][sess["current_idx"]]["category"]})
            if ws_conn:
                await ws_conn.send_json({"type": "evaluation", "evaluation": empty_eval})
            return {"ok": True}

        # Evaluate
        from openrouter_client import OpenRouterClient
        from interview_simulator import _parse_json
        llm = OpenRouterClient()
        q = sess["question_queue"][sess["current_idx"]]

        prompt = f"""Evaluate this interview answer.
Question: {q['question']}
What to look for: {q['what_to_look_for']}
Candidate's answer (raw speech-to-text — may have minor accent/STT errors like wrong word endings or near-homophones; infer the intended technical term from context):
{text}

Return ONLY valid JSON:
{{"score": <int 1-10>, "depth": "<shallow|adequate|deep>",
  "hits": ["<covered>"], "misses": ["<missed>"],
  "needs_follow_up": <true|false>, "follow_up_reason": "<why>",
  "brief_feedback": "<1 sentence>"}}"""

        resp = llm.call_llm([{"role": "user", "content": prompt}], temperature=0.2)
        evaluation = _parse_json(resp)

        # Store
        sess["transcript"].append({
            "question_idx": sess["current_idx"],
            "category": q["category"],
            "question": q["question"],
            "answer": text,
            "evaluation": evaluation,
        })
        sess["answer_scores"].append({"score": evaluation.get("score", 0), "category": q["category"]})

        # Inject follow-up if needed
        needs_fu = evaluation.get("needs_follow_up", False)
        if needs_fu and sess["follow_up_count"] < sess["max_follow_ups"] and q.get("follow_up"):
            fu_q = {**q, "question": q["follow_up"], "follow_up": ""}
            idx = sess["current_idx"]
            sess["question_queue"].insert(idx + 1, fu_q)
            sess["follow_up_count"] += 1
        else:
            sess["follow_up_count"] = 0

        # Push evaluation
        if ws_conn:
            await ws_conn.send_json({"type": "evaluation", "evaluation": evaluation})

        return {"ok": True}

    except Exception as e:
        if ws_conn:
            await ws_conn.send_json({"type": "status", "text": f"Error: {e}", "color": "red"})
        return JSONResponse({"error": str(e)})


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_interview(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"type": "status", "text": "Session not found", "color": "red"})
        return

    sess = sessions[session_id]
    sess["ws"] = websocket

    async def send_question():
        idx = sess["current_idx"]
        queue = sess["question_queue"]
        if idx >= len(queue):
            await finish_interview()
            return

        q = queue[idx]
        await websocket.send_json({
            "type": "question",
            "idx": idx,
            "category": q["category"],
            "question": q["question"],
            "gap": q.get("gap", ""),
        })

        # Generate and send TTS audio
        try:
            import edge_tts, io as _io
            communicate = edge_tts.Communicate(q["question"], voice="en-US-GuyNeural")
            mp3_buf = _io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buf.write(chunk["data"])
            b64 = base64.b64encode(mp3_buf.getvalue()).decode()
            await websocket.send_json({"type": "tts_audio", "audio_b64": b64})
        except Exception:
            pass

    async def finish_interview():
        from openrouter_client import OpenRouterClient
        from interview_simulator import _parse_json
        llm = OpenRouterClient()

        scores = sess["answer_scores"]
        avg = sum(s["score"] for s in scores) / len(scores) if scores else 0

        prompt = f"""Write a final interview assessment.
Candidate: {sess['candidate_name']}
Job: {sess['job_description'][:400]}
Transcript: {json.dumps(sess['transcript'], indent=2)}
Avg score: {avg:.1f}/10

Return ONLY valid JSON:
{{"overall_score": <float>, "verdict": "<HIRE|STRONG_HIRE|HOLD|REJECT>",
  "summary": "<2-3 sentences>",
  "category_scores": {{"technical_depth":<float>,"gap_probing":<float>,"project_specific":<float>,"system_design":<float>,"behavioral":<float>}},
  "top_strengths": [<str>], "key_concerns": [<str>], "recommendation": "<paragraph>"}}"""

        try:
            resp = llm.call_llm([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
            report = _parse_json(resp)
        except Exception as e:
            report = {"overall_score": avg, "verdict": "HOLD", "summary": str(e)}

        await websocket.send_json({"type": "report", "report": report})

        # Persist report to MongoDB
        if is_connected():
            try:
                save_interview_report(
                    session_id=session_id,
                    job_id=sess.get("job_id"),
                    candidate_id=sess["candidate_id"],
                    candidate_name=sess["candidate_name"],
                    job_description=sess["job_description"],
                    transcript=sess["transcript"],
                    report=report,
                )
            except Exception:
                pass

    try:
        async for raw in websocket.iter_json():
            if raw.get("type") == "start":
                await send_question()
            elif raw.get("type") == "next":
                sess["current_idx"] += 1
                if sess["current_idx"] >= len(sess["question_queue"]):
                    await finish_interview()
                else:
                    await send_question()
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
