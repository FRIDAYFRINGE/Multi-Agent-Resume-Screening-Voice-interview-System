# Multi-Agent Resume Screening Assistant

An end-to-end agentic AI pipeline for talent pool screening and live voice interviews.  
Seeds a candidate pool → narrows it against a job description through a 6-node LangGraph pipeline (vector retrieval → cross-encoder rerank → LLM scoring of the shortlist) → conducts real-time voice interviews with TTS questions and Whisper STT answers → produces scored reports with hiring verdicts — all persisted to MongoDB.

---

## Evaluated Performance

Ranking quality is measured with standard IR methodology — TREC-style qrels, jackknife
95% confidence intervals, and a one-sided permutation test — against a 45-candidate
labeled pool across four relevance tiers.

| Metric | Pipeline | Vector-Only | Random |
|---|---|---|---|
| NDCG@10 | **0.984** [0.953, 1.000] | 0.983 [0.813, 1.000] | 0.763 [0.620, 0.905] |
| Recall@10 (Tier-1) | **0.900** [0.708, 1.000] | 0.900 [0.380, 1.000] | 0.400 [0.030, 0.770] |
| Kendall's τ | 0.496 [0.284, 0.709] | 0.499 [0.286, 0.712] | −0.020 [−0.245, 0.206] |
| Spearman's ρ | 0.623 [0.384, 0.862] | 0.620 [0.384, 0.856] | −0.035 [−0.343, 0.273] |

Both the pipeline and a vector-only baseline retrieve the right candidates into the
top 10 (NDCG@10 ≈ 0.98, well clear of random at 0.763). On *full-list* rank
correlation the two are level, and that is a direct consequence of the cost/latency
funnel: only `LLM_TOP_K` (default 15) of 45 candidates receive an LLM score, and the
rest are ordered by the deterministic signals alone.

Widening the funnel trades latency for full-list ordering, and the trade has been
priced:

| Funnel (retrieval / cross-encoder / LLM) | Kendall's τ | Total run |
|---|---:|---:|
| 75 / 25 / 15 *(default)* | 0.496 | **52.5s** |
| 75 / 25 / 25 | 0.536 | 98.8s |
| 75 / 56 / 56 *(whole pool)* | **0.818** | 430.7s |

All three limits are environment variables, so the operating point is a deployment
choice rather than a code change. The default favours interactive latency; raising
`LLM_TOP_K`, `CROSS_ENCODER_TOP_K` and `RETRIEVAL_TOP_K` together recovers full-list
ordering at roughly 8× the runtime.

**Scope of these results.** The labeled pool is synthetic, written to span four
relevance tiers against one job description. It saturates on top-k metrics and it does
not exercise long-document behaviour: no seed exceeds the cross-encoder's 512-token
input budget, whereas typical real resumes do. Ranking quality on real-world resumes
is therefore not established by this benchmark, and the evaluation harness is included
in the repository so the measurement can be repeated on any labeled pool.

*Metrics use TREC-style qrels with jackknife leave-one-out 95% confidence intervals
and a one-sided permutation test (n = 10,000).*

---

## What It Does

1. **Seed a talent pool** — run `seed_candidates.py` once to populate ChromaDB + MongoDB with 45 synthetic candidates across 4 quality tiers (or upload resumes via UI)
2. **Paste a job description** — the pool is narrowed through a three-stage funnel (retrieval → cross-encoder → LLM), the shortlist is evaluated in parallel, and results are ranked and cached by JD fingerprint
3. **Upload new resumes** — parsed, deduplicated by email (one address = one resume, always overwritten), evaluated against the JD:
   - Same JD seen before → fast path: only new resumes re-evaluated, cached pool rankings merged
   - New JD → full pipeline: shortlist scored from scratch
   - Measured: single resume ~96s end to end (parse 50s / evaluate 10s / question generation 35s); 10 resumes ~195s (parses run 6-wide)

   > **Only the shortlist is LLM-scored.** Retrieval keeps `RETRIEVAL_TOP_K` (75), the cross-encoder keeps `CROSS_ENCODER_TOP_K` (25), and only `LLM_TOP_K` (15) candidates reach the LLM. Everyone else is returned ranked by the deterministic signals with `match_score: null`, `overall_recommendation: "NOT_EVALUATED"` and `screening_status: "retrieval_only"` — they appear in the UI as "Not Recommended". All three limits are env-overridable.
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
│              → DeepSeek V3 (OpenRouter) → Resume JSON       │
│              → Identity = email (one address, one resume)   │
│                already known → overwrite in place           │
│                no email      → rejected, not stored         │
│              → chroma_id = slug of the email local-part     │
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
│  [1] Planner      JD → search queries (cached by JD hash)   │
│                   Scoring weights: hardcoded constants       │
│                   SCORE_WEIGHTS = {skills:0.40, exp:0.35,   │
│                                    edu:0.15, proj:0.10}     │
│  [2] Retriever    Full-pool ChromaDB semantic search        │
│                   + cross-encoder CE scores (local, free)  │
│                   + max cosine similarity per candidate     │
│  [3] Evaluator    Parallel LLM scoring (6 workers)          │
│                   Component scores → match_score computed   │
│                   in Python (not LLM) for determinism       │
│  [4] Critique     Read-only QA + flags who to interview     │
│                   No score mutation                          │
│  [5] Synthesizer  Weighted RRF fusion (LLM 50% + CE 30%    │
│                   + vector 20%) → deterministic ranking     │
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

## Determinism & Score Stability

The pipeline is engineered to produce identical scores on repeated runs over the same pool and JD:

| Source of variance | Fix |
|---|---|
| LLM-generated scoring weights | Hardcoded `SCORE_WEIGHTS` constants in `agentic_pipeline.py` |
| Free-form holistic `match_score` | Computed in Python from component scores × fixed weights |
| Planner re-generating search queries | Planner output cached by JD hash in MongoDB `jd_plans` collection |
| Critique mutating scores | Critique is read-only — no `adjustments` loop |
| Non-deterministic evaluation ordering | Evaluations sorted by `candidate_id` before passing to critique |
| Single jittery sort signal | Weighted RRF over 3 signals (LLM + cross-encoder + vector); rank-based fusion absorbs small score wobble |
| Model non-determinism | `deepseek/deepseek-chat-v3-0324` — **36 rank swaps** across 52 candidates, down from 44 with v3.2 (max drift 15 vs 28) |

Verified: 10 parallel runs → `match_score` range = **3.8 pts** (V3-0324), vs 7.1 pts (Flash). Determinism test: **36/52 rank swaps** across two independent full-pipeline runs (16/52 candidates fully stable). Residual variance comes from the LLM evaluator (50% of RRF weight); the cross-encoder and vector signals are fully deterministic and anchor the ranking.

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
│                           candidates / jobs / interview_reports / jd_plans
│                           Email-keyed candidate identity, email dedup,
│                           JD hash deduplication, planner plan cache
│
├── reranker.py             Cross-encoder reranker — ms-marco-MiniLM-L-6-v2
│                           Preloaded at startup (offline-first), lock-guarded
│                           singleton; ce_scores(jd, docs) → 0-1 floats
│                           Note: 512-token input limit truncates long resumes
│
├── agentic_pipeline.py     6-node LangGraph screening pipeline
│                           CandidateVectorStore (ChromaDB)
│                           SCORE_WEIGHTS + RRF_K + FUSION_WEIGHTS constants
│                           rrf_fuse() — weighted RRF over 3 signals
│                           Parallel evaluator (6 workers) + IQ generator (5 workers)
│                           match_score computed in Python, not LLM
│
├── interview_simulator.py  7-node LangGraph interview state machine
│
├── voice_io.py             TTS (edge-tts) + STT (faster-whisper / Groq)
│
├── openrouter_client.py    OpenRouter API wrapper — DeepSeek V3 default
│                           MODEL_V3 / MODEL_FLASH / MODEL_PRO constants
│
├── resume_parser.py        PDF/DOCX/TXT → raw text
│
├── schemas.py              Resume dataclass hierarchy (Python 3.14 compatible)
│
├── seed_candidates.py      Populates ChromaDB + MongoDB with 45 synthetic
│                           candidates across 4 tiers — run once
│
├── chroma_db/              ChromaDB persistent storage (auto-created)
├── requirements.txt
├── .env                    API keys + DB config (create from .env.example)
├── .env.example
└── README.md
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

## Candidate Identity

Email is the only identity: **one address → one candidate → one resume.**

```
john.doe@gmail.com
  MongoDB _id : john.doe@gmail.com   (the address is the primary key)
  chroma_id   : john_doe             (slug of the local-part)

Re-uploading the same address overwrites the stored resume in place.
There are no resume versions, and no name-derived fallback id.

No readable email
  → rejected and reported in parse_errors; never stored under a guess.
```

**Which address is trusted.** The model's answer is used only when that exact
string also appears in the resume text; otherwise the regex-extracted address
wins. Without this check a hallucinated placeholder (`john.doe@example.com`)
silently splits one person into two candidates.

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

# Optional — ChromaDB path (default: ./chroma_db)
CHROMA_DB_PATH=./chroma_db

# Optional — model override (default shown; v3-0324 chosen for determinism)
DEEPSEEK_V3_MODEL=deepseek/deepseek-chat-v3-0324
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

Ingestion is ~97% of cost and is one-time per resume. ChromaDB embeddings are fully local (zero cost). Fast-path re-runs cost near zero. Planner output is cached by JD hash — repeated screens on the same JD cost only the evaluator calls.

---

## Tech Stack

| Layer | Technology |
|---|---|
| File parsing | PyPDF2, python-docx |
| LLM | DeepSeek V3-0324 (`deepseek/deepseek-chat-v3-0324`) via OpenRouter — all pipeline tasks |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free, sentence-transformers) |
| Embeddings | all-MiniLM-L6-v2 (ChromaDB default, ONNX, local, free) |
| Vector DB | ChromaDB 1.5.x, PersistentClient, cosine similarity |
| Persistence | MongoDB 8.3.x, pymongo |
| Orchestration | LangGraph StateGraph (6-node screening + 7-node interview) |
| Parallelism | ThreadPoolExecutor (6 eval workers, 5 IQ workers) |
| Data models | Python dataclasses (Python 3.14 compatible) |
| TTS | edge-tts (Microsoft Edge neural voices, free) |
| STT local | faster-whisper large-v3 (CPU/GPU) |
| STT cloud | Groq Whisper large-v3 API (free tier, ~20× faster) |
| Web API | FastAPI 0.115 + WebSockets |
| Frontend | Vanilla HTML/CSS/JS (dark theme, no framework) |
| Eval metrics | numpy, scipy (NDCG, jackknife CI, permutation test) |
| Runtime | Python 3.14, Windows |

