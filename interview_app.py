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
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from ui import HTML  # noqa: E402  (imported after dotenv)

app = FastAPI(title="AI Interview Assistant")

# ── In-memory session store ────────────────────────────────────────────────────
sessions: dict = {}  # session_id → interview state


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/api/screen")
async def screen_endpoint(
    resume: UploadFile = File(...),
    jd: str = Form(""),
    skills: str = Form(""),
    max_follow_ups: int = Form(1)
):
    from resume_parser import ResumeParser
    from openrouter_client import OpenRouterClient
    from agentic_pipeline import ScreeningPipeline, ScreeningQuery
    from interview_simulator import _flatten_questions

    try:
        # Save uploaded file
        suffix = os.path.splitext(resume.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await resume.read())
            tmp_path = tmp.name

        llm = OpenRouterClient()
        pipeline = ScreeningPipeline()

        parsed = ResumeParser().parse_resume(tmp_path, llm)
        os.unlink(tmp_path)

        cid = (parsed.contact_info.name or "candidate").lower().replace(" ", "_")
        pipeline.add_candidate(parsed, cid)

        required_skills = [s.strip() for s in skills.split(",") if s.strip()]
        state = pipeline.screen_candidates(ScreeningQuery(
            job_description=jd,
            required_skills=required_skills,
            min_experience_years=1,
        ))

        iq = state["interview_questions"].get(cid)
        if not iq:
            top = state["final_rankings"][0] if state["final_rankings"] else {}
            return JSONResponse({"error": f"Not recommended for interview. Score: {top.get('match_score', 0)}/100"})

        session_id = str(uuid.uuid4())
        question_queue = _flatten_questions(iq)

        sessions[session_id] = {
            "candidate_id": cid,
            "candidate_name": parsed.contact_info.name or cid,
            "job_description": jd,
            "question_queue": question_queue,
            "current_idx": 0,
            "follow_up_count": 0,
            "max_follow_ups": max_follow_ups,
            "transcript": [],
            "answer_scores": [],
            "match_score": iq.get("match_score"),
            "evaluation_map": {r["candidate_id"]: r for r in state["final_rankings"]},
            "iq": iq,
        }

        return {
            "session_id": session_id,
            "candidate_name": parsed.contact_info.name or cid,
            "match_score": iq.get("match_score"),
            "total_questions": len(question_queue),
        }

    except Exception as e:
        return JSONResponse({"error": str(e)})


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

        # Transcribe
        from voice_io import SpeechToText
        stt = sess.get("stt")
        if not stt:
            stt = SpeechToText(model_size="large-v3")
            sess["stt"] = stt

        # initial_prompt primes vocabulary — add question text so Whisper expects domain terms
        q_text = sess["question_queue"][sess["current_idx"]].get("question", "")
        TECH_PROMPT = (
            "JD parsing, job description, routing logic, deterministic routing, "
            "schema validation, field extraction, fallback strategy, rule-based parser, "
            "LangGraph, ChromaDB, FAISS, BM25, RAG pipeline, semantic search, "
            "dense embeddings, Reciprocal Rank Fusion, RRF, reranking, chunking, "
            "PyTorch, FastAPI, LoRA, QLoRA, RLHF, fine-tuning, vector database, "
            "cosine similarity, LLM, transformer, hyperparameter, agentic pipeline, "
            "end-to-end latency, throughput, Docker, CI/CD. "
            f"Question: {q_text}"
        )
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
