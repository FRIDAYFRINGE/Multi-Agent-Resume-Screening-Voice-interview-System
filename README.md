# Multi-Agent Resume Screening Assistant

An end-to-end agentic AI pipeline for talent pool screening and live voice interviews.  
Seeds a candidate pool → screens the entire pool against a job description with a 6-node LangGraph pipeline → conducts real-time voice interviews with TTS questions and Whisper STT answers → produces scored reports with hiring verdicts — all persisted to MongoDB.

---

## What It Does

1. **Seed a talent pool** — run `seed_candidates.py` once to populate ChromaDB + MongoDB with candidates (or upload resumes via UI)
2. **Paste a job description** — the full pool is evaluated in parallel; results are ranked and cached by JD fingerprint
3. **Upload new resumes** — parsed, deduplicated by filename + content similarity, evaluated against the JD:
   - Same JD seen before → fast path (~10-15s): only new resumes re-evaluated, cached pool rankings merged
   - New JD → full pipeline (~90s): all candidates scored from scratch
4. **Interview recommended candidates** — Strong/Good Fit candidates get tailored question banks; live voice interview in browser with TTS + Whisper STT
5. **Final report** — verdict (HIRE / STRONG HIRE / HOLD / REJECT), per-category scores, strengths, concerns, full transcript — saved to MongoDB

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your keys (see Environment Variables below)

# 3. Seed the talent pool (run once)
python seed_candidates.py

# 4. Start the server
python interview_app.py
```

Open **http://localhost:8001** in your browser.

MongoDB is optional — the app works without it (ChromaDB only), but persistence, history, and JD deduplication require a running MongoDB instance (`mongodb://localhost:27017` by default).

---

## STT Provider

Switch between local and cloud transcription with one line in `interview_app.py`:

```python
STT_PROVIDER = "local"   # faster-whisper large-v3, runs on CPU, free, slow
STT_PROVIDER = "groq"    # Groq Whisper large-v3 API, free tier, ~20x faster
```

For Groq: get a free key at https://console.groq.com/keys and add `GROQ_API_KEY=...` to `.env`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INGESTION                                                   │
│  PDF/DOCX/TXT → ResumeParser → raw text                     │
│              → OpenRouter LLM → Resume dataclass (JSON)     │
│              → Filename dedup (MongoDB resumes[] array)     │
│                ≥90% similar  → skip (reuse sub_id)          │
│                <90% similar  → overwrite in-place           │
│                new filename  → new sub_id                   │
│              → chroma_id = base_id + "_" + sub_id           │
│              → ChromaDB (all-MiniLM-L6-v2, cosine)          │
│              → MongoDB candidates collection                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  JD DEDUPLICATION                                            │
│  SHA-256[:16] hash → find_job_by_hash()                     │
│  Same JD + uploads → FAST PATH                              │
│    Evaluate uploads only → merge cached pool → sessions     │
│  New JD → FULL PIPELINE                                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  SCREENING  (LangGraph StateGraph — 6 nodes)                 │
│  [1] Planner      JD → search queries + scoring weights     │
│  [2] Retriever    Full-pool ChromaDB search (no cap)        │
│  [3] Evaluator    Parallel LLM scoring (6 workers)          │
│  [4] Critique     Validates scores, flags who to interview  │
│  [5] Synthesizer  Final ranked list + match_level labels    │
│  [6] IQ Generator Parallel tailored questions (top 5)       │
│  → Saved to MongoDB jobs collection with jd_hash            │
└─────────────────────────────┬───────────────────────────────┘
                              │  sessions for:
                              │  • top-5 recommended pool candidates
                              │  • all uploaded Strong/Good Fit
┌─────────────────────────────▼───────────────────────────────┐
│  INTERVIEW  (LangGraph StateGraph — 7 nodes + WebSocket)     │
│  load_questions → ask_question → listen → evaluate          │
│       ↑               │ TTS (edge-tts)                      │
│       └── decide ◄────┘ STT (Whisper local or Groq API)     │
│                      → report (verdict + scores)            │
│  → Saved to MongoDB interview_reports collection            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  WEB UI  (FastAPI + HTML/JS — http://localhost:8001)         │
│  GET  /api/pool-count        candidate count + mongo status │
│  GET  /api/jobs              past screening runs history    │
│  GET  /api/jobs/{id}         full past run (read-only)      │
│  GET  /api/reports           completed interview reports    │
│  POST /api/screen            upload + JD → pipeline/fast   │
│  POST /api/submit-answer     audio → STT → eval → WS push  │
│  WS   /ws/{session_id}       question flow + TTS + results  │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
interview_agentic/
│
├── interview_app.py        Main entry point — FastAPI + WebSocket server
│                           STT_PROVIDER variable (top of file)
│                           Fast path + full pipeline logic
│                           Auto-interview for uploaded Good/Strong Fit
│
├── ui.py                   Full dark-theme HTML/CSS/JS frontend
│                           "Your Uploaded Candidates" highlight card
│                           Full talent pool rankings table
│                           Cached-run banner, past runs history
│
├── database.py             MongoDB integration (pymongo)
│                           candidates / jobs / interview_reports
│                           Candidate ID scheme, filename dedup,
│                           JD hash deduplication
│
├── agentic_pipeline.py     6-node LangGraph screening pipeline
│                           CandidateVectorStore (ChromaDB)
│                           Parallel evaluator + IQ generator
│                           Full-pool retrieval
│
├── interview_simulator.py  7-node LangGraph interview state machine
│
├── voice_io.py             TTS (edge-tts) + STT (faster-whisper / Groq)
│
├── openrouter_client.py    OpenRouter API wrapper (DeepSeek v4 Flash)
│
├── resume_parser.py        PDF/DOCX/TXT → raw text
│
├── schemas.py              Resume dataclass hierarchy (no Pydantic)
│
├── seed_candidates.py      Populates ChromaDB + MongoDB with 49 synthetic
│                           candidates across 4 tiers — run once
│
├── chroma_db/              ChromaDB persistent storage (auto-created)
├── requirements.txt
├── .env                    API keys + DB config
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

## Candidate ID Scheme

```
Email: john.doe@gmail.com
  base_id   : john_doe
  doc_key   : john.doe@gmail.com  (MongoDB _id)
  chroma_id : john_doe_1  (first resume)
              john_doe_2  (second resume, different filename)

No email:
  base_id   : john_doe  (slugified name)
  doc_key   : john_doe  (MongoDB _id)
  chroma_id : john_doe_1, john_doe_2, ...

Filename deduplication (same email, same filename):
  ≥ 90% Jaccard similarity → skip, reuse existing chroma_id
  <  90% Jaccard similarity → overwrite that sub_id in-place
```

---

## Environment Variables

```bash
# Required
OPENROUTER_API_KEY=sk-or-...       # openrouter.ai

# Optional — for fast cloud STT
GROQ_API_KEY=gsk_...               # console.groq.com/keys

# Optional — for MongoDB persistence
MONGO_URI=mongodb://localhost:27017
MONGO_DB=interview_agentic

# Optional — defaults to ./chroma_db
CHROMA_DB_PATH=./chroma_db

# Optional — model override
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

---

## Cost Estimate (DeepSeek V3 via OpenRouter)

| Scenario | Cost |
|---|---|
| 1 JD × 1,000 resumes (first parse + screen) | ~$0.39 |
| 1 JD × 1,000 resumes (already parsed, new JD) | ~$0.01 |
| 10 JDs × 1,000 resumes (parse once) | ~$0.49 |
| Same JD re-run + 1 new upload (fast path) | ~$0.001 |
| 1 JD × 10,000 resumes | ~$3.64 |

Ingestion is ~97% of cost and is one-time per resume. ChromaDB embeddings are fully local (zero cost). Fast-path re-runs cost near zero.

---

## Tech Stack

| Layer | Technology |
|---|---|
| File parsing | PyPDF2, python-docx |
| LLM | DeepSeek v4 Flash via OpenRouter |
| Embeddings | all-MiniLM-L6-v2 (ChromaDB default, ONNX, local) |
| Vector DB | ChromaDB 1.5.x, PersistentClient, cosine similarity |
| Persistence | MongoDB 8.3.x, pymongo |
| Orchestration | LangGraph StateGraph (6-node + 7-node) |
| Parallelism | ThreadPoolExecutor (6 eval workers, 5 IQ workers) |
| Data models | Python dataclasses (Python 3.14 compatible) |
| TTS | edge-tts (Microsoft Edge neural voices, free) |
| STT local | faster-whisper large-v3 (CPU/GPU) |
| STT cloud | Groq Whisper large-v3 API (free tier, ~20x faster) |
| Web API | FastAPI 0.115 + WebSockets |
| Frontend | Vanilla HTML/CSS/JS (dark theme, no framework) |
| Runtime | Python 3.14, Windows |

---

## Contact

this.vishalchuhan@gmail.com
