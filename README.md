# Multi-Agent Resume Screening Assistant

An end-to-end agentic AI pipeline for resume screening and live voice interviews.  
Parses resumes → screens candidates with a 6-node LangGraph pipeline → conducts a real-time voice interview with TTS questions and Whisper STT answers → produces a scored report with hiring verdict.

---

## What It Does

1. **Upload** a resume (PDF/DOCX/TXT) and a job description
2. **Screening pipeline** runs automatically:
   - LLM extracts all resume fields into structured JSON
   - ChromaDB indexes the candidate with sentence embeddings
   - 6 LangGraph agents plan, retrieve, evaluate, critique, rank, and generate tailored interview questions
3. **Live voice interview** starts in the browser:
   - Questions are spoken aloud via Microsoft Edge TTS
   - Candidate records answers (microphone)
   - Whisper (local or Groq cloud) transcribes each answer
   - LLM evaluates the answer and optionally asks a follow-up
4. **Final report** with overall score, verdict (HIRE / STRONG HIRE / HOLD / REJECT), per-category scores, strengths, concerns, and full transcript

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your OPENROUTER_API_KEY (required)
# Add your GROQ_API_KEY (optional — for fast cloud STT)

# 3. Run
python interview_app.py
```

Open **http://localhost:8001** in your browser.

---

## STT Provider

Switch between local and cloud transcription with one line in `interview_app.py`:

```python
STT_PROVIDER = "local"   # faster-whisper large-v3, runs on CPU, free, slow (~1x real-time)
STT_PROVIDER = "groq"    # Groq Whisper large-v3 API, free tier, ~20x faster
```

For Groq: get a free key at https://console.groq.com/keys and add `GROQ_API_KEY=...` to `.env`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INGESTION                                                   │
│  PDF/DOCX/TXT → ResumeParser (PyPDF2) → raw text            │
│              → OpenRouter LLM → Resume dataclass (JSON)     │
│              → ChromaDB (all-MiniLM-L6-v2, cosine)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  SCREENING  (LangGraph StateGraph — 6 nodes)                 │
│  [1] Planner      JD → search queries + scoring weights     │
│  [2] Retriever    ChromaDB semantic search + dedup/rerank   │
│  [3] Evaluator    LLM scores each candidate vs JD           │
│  [4] Critique     Validates scores, flags who to interview  │
│  [5] Synthesizer  Final ranked list                         │
│  [6] IQ Generator Per-candidate tailored question bank      │
└─────────────────────────────┬───────────────────────────────┘
                              │  (recommended candidates only)
┌─────────────────────────────▼───────────────────────────────┐
│  INTERVIEW  (LangGraph StateGraph — 7 nodes + WebSocket)     │
│  load_questions → ask_question → listen → evaluate          │
│       ↑               │ TTS audio (edge-tts)                │
│       └── decide ◄────┘ STT (Whisper local or Groq API)     │
│       (follow-up or next question)                          │
│                      → report (verdict + scores)            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  WEB UI  (FastAPI + HTML/JS — http://localhost:8001)         │
│  POST /api/screen        upload resume + JD → run pipeline  │
│  POST /api/submit-answer audio upload → STT → eval → WS     │
│  WS   /ws/{session_id}   question flow + TTS + eval results │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
interview_agentic/
│
├── interview_app.py        Main entry point — FastAPI server
│                           STT_PROVIDER variable (top of file)
│                           POST /api/screen, /api/submit-answer
│                           WebSocket /ws/{session_id}
│
├── ui.py                   Full dark-theme HTML/CSS/JS frontend
│                           (imported by interview_app.py)
│
├── interview_simulator.py  LangGraph interview state machine
│                           InterviewState, 7-node graph
│                           _flatten_questions(), _parse_json()
│
├── voice_io.py             TTS + STT classes
│                           TextToSpeech  — edge-tts (free, no API key)
│                           SpeechToText  — faster-whisper (local)
│                           GroqSTT       — Groq Whisper API (cloud)
│
├── agentic_pipeline.py     Screening pipeline
│                           PipelineState (TypedDict)
│                           CandidateVectorStore (ChromaDB)
│                           AgentNodes (all 6 node methods)
│                           ScreeningPipeline (facade)
│
├── openrouter_client.py    OpenRouter API wrapper
│                           call_llm(), extract_resume_from_text()
│                           evaluate_candidate(), _dict_to_resume()
│
├── resume_parser.py        File → raw text extraction
│                           PDF via PyPDF2, DOCX via python-docx
│
├── schemas.py              Resume dataclass hierarchy
│                           ContactInfo, Experience, Education,
│                           Skill, Project, Certification, Resume
│                           (Python dataclasses — no Pydantic)
│
├── run_interview.py        CLI entry point (terminal-only)
│
├── main.py                 Legacy FastAPI endpoints (screening only)
│
├── chroma_db/              ChromaDB persistent storage (auto-created)
├── requirements.txt
├── .env                    OPENROUTER_API_KEY, GROQ_API_KEY
├── .env.example
├── TODO.md
└── ARCHITECTURE.md
```

---

## Resume Fields Extracted

| Section | Fields |
|---|---|
| Contact | name, email, phone, address, city, state, country, linkedin, github, portfolio, website |
| Profile | professional_summary |
| Skills | category + skills[] (grouped) |
| Experience | company, position, employment_type, dates, location, description, achievements[], technologies[] |
| Education | institution, degree, field_of_study, dates, gpa, activities |
| Certifications | title, issuer, date_obtained, expiration, url |
| Projects | title, description, technologies[], url |
| Other | languages, publications, volunteer_experience, awards_recognition |

---

## Interview Question Categories

| Category | Description |
|---|---|
| Technical Depth | Tests depth of key skills from resume |
| Gap Probing | Targets specific weaknesses vs JD |
| Project-Specific | Questions tied to named projects in resume |
| System Design | Role-relevant architecture questions |
| Behavioral | Grounded in actual resume gaps and context |

Each question includes `what_to_look_for` and a `follow_up` for shallow answers.

---

## Environment Variables

```bash
# Required
OPENROUTER_API_KEY=sk-or-...       # openrouter.ai
OPENROUTER_MODEL=deepseek/deepseek-v4-flash

# Required for ChromaDB path
CHROMA_DB_PATH=./chroma_db

# Optional — only needed if STT_PROVIDER="groq"
GROQ_API_KEY=gsk_...               # console.groq.com/keys
```

---

## Cost Estimate (DeepSeek V3 via OpenRouter)

| Scenario | Cost |
|---|---|
| 1 JD × 1,000 resumes (first parse + screen) | ~$0.39 |
| 1 JD × 1,000 resumes (already parsed, new JD) | ~$0.01 |
| 10 JDs × 1,000 resumes (parse once) | ~$0.49 |
| 1 JD × 10,000 resumes | ~$3.64 |

Ingestion is ~97% of cost and is one-time per resume. ChromaDB embeddings are fully local (zero cost).

---

## Tech Stack

| Layer | Technology |
|---|---|
| File parsing | PyPDF2, python-docx |
| LLM | DeepSeek v4 Flash via OpenRouter |
| Embeddings | all-MiniLM-L6-v2 (ChromaDB default, ONNX, local) |
| Vector DB | ChromaDB 1.5.x, PersistentClient, cosine similarity |
| Orchestration | LangGraph StateGraph |
| Data models | Python dataclasses (Python 3.14 compatible) |
| TTS | edge-tts (Microsoft Edge neural voices, free) |
| STT local | faster-whisper large-v3 (CPU/GPU) |
| STT cloud | Groq Whisper large-v3 API (free tier) |
| Web API | FastAPI 0.115 + WebSockets |
| Frontend | Vanilla HTML/CSS/JS (dark theme, no framework) |
| Runtime | Python 3.14, Windows |
