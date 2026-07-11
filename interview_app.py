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
from openrouter_client import MODEL_V3
from database import (save_candidate, save_job, save_interview_report,
                      list_jobs, get_job, list_reports, is_connected,
                      resolve_candidate_id, add_resume_version,
                      find_resume_by_filename, update_resume_version_inplace,
                      make_jd_hash, find_job_by_hash,
                      save_session, update_session_transcript,
                      mark_session_completed, load_session,
                      get_report_by_session)

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
    resp = llm.call_llm([{"role": "user", "content": prompt}], model=MODEL_V3, temperature=0.1)
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


def _raw_text_similarity(raw_text: str, resume_dict: dict) -> float:
    """Jaccard similarity between raw resume text and a stored resume JSON (0.0 – 1.0)."""
    import re as _re
    wa = set(_re.findall(r'\w+', (raw_text or "").lower()))
    wb = set(_re.findall(r'\w+', json.dumps(resume_dict or {}).lower()))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _extract_email_from_text(raw_text: str) -> str:
    """Regex email extraction from raw resume text — no LLM needed."""
    import re as _re
    m = _re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', raw_text or "")
    return m.group().strip().lower() if m else ""


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
        resp = llm.call_llm([{"role": "user", "content": prompt}], model=MODEL_V3, temperature=0.1)
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
                    # ── Local text extraction (fast, no LLM) ──────────────────
                    sfx = suffix.lower()
                    if sfx == ".pdf":
                        raw_text = parser.extract_text_from_pdf(tmp_path)
                    elif sfx == ".docx":
                        raw_text = parser.extract_text_from_docx(tmp_path)
                    else:
                        raw_text = parser.extract_text_from_txt(tmp_path)

                    # ── Pre-dedup: skip the LLM parse for re-uploaded files ───
                    # Same filename + ≥85% raw-text similarity → reuse existing
                    # entry without spending an LLM call on parsing.
                    raw_email = _extract_email_from_text(raw_text)
                    if raw_email and is_connected() and upload.filename:
                        pre_entry = find_resume_by_filename(raw_email, upload.filename)
                        if pre_entry:
                            ex_cid    = pre_entry["chroma_id"]
                            ex_resume = pipeline.store.get_candidate_data(ex_cid) or {}
                            if ex_resume and _raw_text_similarity(raw_text, ex_resume) >= 0.85:
                                uploaded_cids.append(ex_cid)
                                continue

                    # ── LLM parse (only for new or changed resumes) ───────────
                    parsed = llm.extract_resume_from_text(raw_text)
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

            # Local ranking signals for the uploads (cross-encoder + vector cosine)
            try:
                import reranker
                _search_hits = {h["id"]: h["similarity_score"]
                                for h in pipeline.store.search(jd, top_k=pipeline.store.count())}
                _docs_result = pipeline.store.collection.get(
                    ids=[e["candidate_id"] for e in uploaded_evals], include=["documents"])
                _doc_by_id = dict(zip(_docs_result.get("ids", []), _docs_result.get("documents", []) or []))
                _ce = reranker.ce_scores(jd, [_doc_by_id.get(e["candidate_id"], "") for e in uploaded_evals])
                for e, s in zip(uploaded_evals, _ce):
                    e["ce_score"]          = round(s, 4)
                    e["vector_similarity"] = _search_hits.get(e["candidate_id"])
            except Exception:
                pass  # signals missing → legacy sort fallback below

            # Merge: new evals on top of cached pool rankings
            cached_rankings = [
                r for r in (existing_job.get("rankings") or [])
                if r.get("candidate_id") not in set(uploaded_cids)  # avoid dupe if re-uploaded
            ]
            merged = uploaded_evals + cached_rankings

            # RRF fusion when every entry carries the deterministic signals;
            # legacy cached runs (pre-upgrade, no ce_score stored) fall back to
            # plain match_score ordering.
            if merged and all(r.get("ce_score") is not None for r in merged):
                from agentic_pipeline import rrf_fuse
                merged = rrf_fuse(merged)
            else:
                merged.sort(key=lambda x: (-x.get("match_score", 0), str(x.get("candidate_id", ""))))
                for i, r in enumerate(merged):
                    r["rank"] = i + 1

            for r in merged:
                r["match_level"] = _match_level(
                    r.get("match_score", 0), r.get("overall_recommendation", "")
                )
                if r.get("candidate_id") in set(uploaded_cids):
                    r["newly_uploaded"] = True

            # Create sessions upfront only for uploaded Strong/Good Fit (parallel IQ).
            # Pool candidates get lazy IQ via POST /api/prepare-interview on click.
            sessions_out: dict = {}
            needs_iq = [
                r for r in merged
                if r.get("newly_uploaded")
                and r.get("match_level") in ("Strong Fit", "Good Fit")
            ]

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
                if is_connected():
                    save_session(sid, sessions[sid])
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
            if is_connected():
                save_session(sid, sessions[sid])
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
                if is_connected():
                    save_session(sid, sessions[sid])
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
            "job_id":        job_id,
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
        def _local_transcribe(path: str) -> str:
            from voice_io import SpeechToText
            local_stt = sess.get("local_stt")
            if not local_stt:
                local_stt = SpeechToText(model_size="large-v3")
                sess["local_stt"] = local_stt
            segs, _ = local_stt.model.transcribe(
                path, language="en", beam_size=5,
                initial_prompt=TECH_PROMPT, vad_filter=True,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segs).strip()

        if STT_PROVIDER == "groq":
            from voice_io import GroqSTT
            stt = sess.get("stt")
            if not stt:
                try:
                    stt = GroqSTT()
                    sess["stt"] = stt
                except ValueError:
                    stt = None  # no API key — fall through to local
            try:
                if stt is None:
                    raise RuntimeError("no key")
                text = stt.transcribe_file(audio_path, prompt=TECH_PROMPT)
            except Exception as groq_err:
                err_str = str(groq_err)
                if "403" in err_str:
                    reason = "Groq daily limit reached or key invalid"
                elif "401" in err_str:
                    reason = "Groq API key rejected"
                else:
                    reason = f"Groq error ({err_str[:60]})"
                if ws_conn:
                    await ws_conn.send_json({
                        "type": "status",
                        "text": f"{reason} — switching to local Whisper",
                        "color": "yellow",
                    })
                text = _local_transcribe(audio_path)
            finally:
                os.unlink(audio_path)
        else:
            text = _local_transcribe(audio_path)
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
            if is_connected():
                update_session_transcript(
                    session_id, sess["current_idx"], sess["follow_up_count"],
                    sess["transcript"], sess["answer_scores"],
                )
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

        resp = llm.call_llm([{"role": "user", "content": prompt}], model=MODEL_V3, temperature=0.1)
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

        # Persist updated transcript to MongoDB
        if is_connected():
            update_session_transcript(
                session_id,
                sess["current_idx"],
                sess["follow_up_count"],
                sess["transcript"],
                sess["answer_scores"],
            )

        # Push evaluation
        if ws_conn:
            await ws_conn.send_json({"type": "evaluation", "evaluation": evaluation})

        return {"ok": True}

    except Exception as e:
        if ws_conn:
            await ws_conn.send_json({"type": "status", "text": f"Error: {e}", "color": "red"})
        return JSONResponse({"error": str(e)})


# ── Report endpoints ──────────────────────────────────────────────────────────

@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Full interview report JSON for a completed session."""
    if not is_connected():
        return JSONResponse({"error": "MongoDB not available"}, status_code=503)
    doc = get_report_by_session(session_id)
    if not doc:
        return JSONResponse({"error": "Report not found"}, status_code=404)
    doc["_id"] = str(doc["_id"])
    if doc.get("completed_at"):
        doc["completed_at"] = doc["completed_at"].isoformat()
    return doc


def _generate_pdf(doc: dict) -> bytes:
    """Build an interview report PDF using fpdf2."""
    from fpdf import FPDF

    def _safe(text: str) -> str:
        """Strip characters outside Latin-1 so Helvetica can render them."""
        return (text or "").encode("latin-1", errors="replace").decode("latin-1")

    VERDICT_COLORS = {
        "STRONG_HIRE": (34, 197, 94),
        "HIRE":        (59, 130, 246),
        "HOLD":        (234, 179, 8),
        "REJECT":      (239, 68, 68),
    }
    GRAY   = (107, 114, 128)
    DARK   = (17, 24, 39)
    LIGHT  = (249, 250, 251)
    ACCENT = (99, 102, 241)

    report   = doc.get("report") or {}
    verdict  = (report.get("verdict") or "HOLD").upper()
    v_color  = VERDICT_COLORS.get(verdict, VERDICT_COLORS["HOLD"])
    name     = _safe(doc.get("candidate_name") or "Candidate")
    jd_snip  = _safe((doc.get("job_description") or "")[:120].replace("\n", " "))
    overall  = report.get("overall_score") or 0
    summary  = _safe(report.get("summary") or "")
    cat_scores = report.get("category_scores") or {}
    strengths  = [_safe(s) for s in (report.get("top_strengths") or [])]
    concerns   = [_safe(c) for c in (report.get("key_concerns") or [])]
    transcript = doc.get("transcript") or []
    completed  = doc.get("completed_at")
    date_str   = completed.strftime("%d %b %Y") if hasattr(completed, "strftime") else str(completed or "")[:10]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # ── Header bar ────────────────────────────────────────────────────────────
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Interview Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 6, date_str, ln=True, align="C")

    # ── Candidate name + verdict badge ────────────────────────────────────────
    pdf.set_y(34)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 10, name, ln=False)
    # Verdict badge (right-aligned)
    pdf.set_fill_color(*v_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(155)
    pdf.cell(35, 10, verdict.replace("_", " "), border=0, ln=True, align="C", fill=True)

    # JD snippet
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 5, f"Role: {jd_snip}", ln=True)
    pdf.ln(2)

    # ── Overall score ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 14, f"{overall:.1f} / 10", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "Overall Interview Score", ln=True, align="C")
    pdf.ln(4)

    # ── Divider ───────────────────────────────────────────────────────────────
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary:
        pdf.set_x(20)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, "Summary", ln=True)
        pdf.set_x(20)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(170, 6, summary)
        pdf.ln(3)

    # ── Category scores table ─────────────────────────────────────────────────
    if cat_scores:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, "Category Scores", ln=True)
        pdf.set_font("Helvetica", "", 10)
        col_w = 85
        for i, (cat, score) in enumerate(cat_scores.items()):
            if i % 2 == 0:
                pdf.set_x(20)
            label = cat.replace("_", " ").title()
            score_val = f"{score:.1f}" if isinstance(score, float) else str(score)
            # Score bar
            bar_max = 80
            bar_len = int((float(score_val) / 10) * bar_max) if score_val else 0
            x0, y0 = pdf.get_x(), pdf.get_y()
            pdf.set_text_color(50, 50, 50)
            pdf.cell(col_w - 20, 6, f"{label}: {score_val}/10", ln=False)
            # Mini bar
            pdf.set_fill_color(*ACCENT)
            pdf.rect(x0 + col_w - 20, y0 + 2, bar_len * 0.25, 3, "F")
            if i % 2 == 1:
                pdf.ln(7)
            else:
                pdf.set_x(20 + col_w)
        pdf.ln(5)

    # ── Strengths & Concerns ──────────────────────────────────────────────────
    half_w = 82
    y_sc = pdf.get_y()

    if strengths:
        pdf.set_xy(20, y_sc)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(34, 197, 94)
        pdf.cell(half_w, 6, "Top Strengths", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        for s in strengths[:4]:
            pdf.set_x(20)
            pdf.multi_cell(half_w, 5, f"+ {s}")

    if concerns:
        pdf.set_xy(110, y_sc)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(239, 68, 68)
        pdf.cell(half_w, 6, "Key Concerns", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        for c in concerns[:4]:
            pdf.set_x(110)
            pdf.multi_cell(half_w, 5, f"- {c}")

    pdf.ln(8)

    # ── Transcript ────────────────────────────────────────────────────────────
    if transcript:
        pdf.set_draw_color(*ACCENT)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 8, "Interview Transcript", ln=True)

        for i, entry in enumerate(transcript):
            cat   = _safe(entry.get("category", "").replace("_", " ").title())
            q     = _safe(entry.get("question", ""))
            ans   = _safe(entry.get("answer", ""))
            ev    = entry.get("evaluation") or {}
            score = ev.get("score", "-")
            depth = _safe(ev.get("depth", ""))
            fb    = _safe(ev.get("brief_feedback", ""))

            # Question header
            pdf.set_x(20)
            pdf.set_fill_color(*LIGHT)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*ACCENT)
            pdf.cell(0, 6, f"Q{i+1} - {cat}   Score: {score}/10  ({depth})", ln=True, fill=True)

            pdf.set_x(20)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(170, 5, q)

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 60, 60)
            if ans:
                pdf.set_x(20)
                pdf.multi_cell(170, 5, f"Answer: {ans[:400]}")
            if fb:
                pdf.set_x(20)
                pdf.set_text_color(*GRAY)
                pdf.multi_cell(170, 5, f"Feedback: {fb}")
            pdf.ln(3)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, "Generated by AI Interview Assistant", align="C")

    return bytes(pdf.output())


@app.get("/api/report/{session_id}/pdf")
async def download_report_pdf(session_id: str):
    """Download the interview report as a formatted PDF."""
    from fastapi.responses import Response
    if not is_connected():
        return JSONResponse({"error": "MongoDB not available"}, status_code=503)
    doc = get_report_by_session(session_id)
    if not doc:
        return JSONResponse({"error": "Report not found"}, status_code=404)
    try:
        pdf_bytes = _generate_pdf(doc)
    except Exception as e:
        return JSONResponse({"error": f"PDF generation failed: {e}"}, status_code=500)
    safe_name = (doc.get("candidate_name") or session_id).replace(" ", "_")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="interview_{safe_name}.pdf"'},
    )


@app.post("/api/prepare-interview")
async def prepare_interview(
    candidate_id: str = Form(""),
    job_id: str = Form(""),
    match_score: float = Form(0),
    max_follow_ups: int = Form(1),
):
    """Generate interview questions + create a session on demand (lazy IQ)."""
    from agentic_pipeline import CandidateVectorStore
    from interview_simulator import _flatten_questions
    from openrouter_client import OpenRouterClient

    store = CandidateVectorStore(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    resume_dict = store.get_candidate_data(candidate_id)
    if not resume_dict:
        return JSONResponse({"error": "Candidate not found"}, status_code=404)

    job = None
    if is_connected() and job_id:
        try:
            job = get_job(job_id)
        except Exception:
            job = None
    jd = (job or {}).get("job_description", "")
    if not jd:
        return JSONResponse({"error": "Job not found — cannot generate questions without a JD"}, status_code=404)

    ranking = next(
        (r for r in (job.get("rankings") or []) if r.get("candidate_id") == candidate_id),
        {"match_score": match_score, "strengths": [], "weaknesses": [],
         "matched_skills": [], "missing_skills": []},
    )

    llm = OpenRouterClient()
    try:
        iq = _generate_iq(llm, jd, ranking, resume_dict)
    except Exception as e:
        return JSONResponse({"error": f"Question generation failed: {e}"}, status_code=500)

    iq["candidate_id"] = candidate_id
    iq["match_score"]  = ranking.get("match_score", match_score)
    sid = str(uuid.uuid4())
    question_queue = _flatten_questions(iq)
    candidate_name = (resume_dict.get("contact_info") or {}).get("name") or candidate_id

    sessions[sid] = {
        "candidate_id":    candidate_id,
        "candidate_name":  candidate_name,
        "job_description": jd,
        "job_id":          job_id,
        "question_queue":  question_queue,
        "current_idx":     0,
        "follow_up_count": 0,
        "max_follow_ups":  max_follow_ups,
        "transcript":      [],
        "answer_scores":   [],
        "match_score":     ranking.get("match_score", match_score),
        "iq":              iq,
    }
    if is_connected():
        save_session(sid, sessions[sid])

    return {
        "session_id":      sid,
        "total_questions": len(question_queue),
        "candidate_name":  candidate_name,
    }


@app.post("/api/session/{session_id}/restart")
async def restart_session(session_id: str):
    """Reset a session back to Q1 without re-running screening or IQ generation."""
    from interview_simulator import _flatten_questions
    if session_id not in sessions:
        if is_connected():
            stored = load_session(session_id)
            if stored:
                sessions[session_id] = stored
    if session_id not in sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    sess = sessions[session_id]
    sess["current_idx"]     = 0
    sess["follow_up_count"] = 0
    sess["transcript"]      = []
    sess["answer_scores"]   = []
    sess["question_queue"]  = _flatten_questions(sess["iq"])  # restore original, removes injected follow-ups
    if is_connected():
        update_session_transcript(session_id, 0, 0, [], [])
    return {"ok": True, "total_questions": len(sess["question_queue"])}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_interview(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        # Try restoring from MongoDB (e.g. after server restart)
        if is_connected():
            stored = load_session(session_id)
            if stored:
                sessions[session_id] = stored
            else:
                await websocket.send_json({"type": "status", "text": "Session not found", "color": "red"})
                return
        else:
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
            resp = llm.call_llm([{"role": "user", "content": prompt}], model=MODEL_V3, temperature=0.1)
            report = _parse_json(resp)
        except Exception as e:
            report = {"overall_score": avg, "verdict": "HOLD", "summary": str(e)}

        await websocket.send_json({"type": "report", "report": report})

        # Persist report to MongoDB + mark session completed
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
                mark_session_completed(session_id)
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
