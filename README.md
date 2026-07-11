# Multi-Agent Resume Screening Assistant

An end-to-end agentic AI pipeline for talent pool screening and live voice interviews.  
Seeds a candidate pool → screens the entire pool against a job description with a 6-node LangGraph pipeline → conducts real-time voice interviews with TTS questions and Whisper STT answers → produces scored reports with hiring verdicts — all persisted to MongoDB.

---

## Evaluated Performance

> **Kendall's τ = 0.825 (95% CI 0.779–0.870)** on a 45-candidate, 4-tier labeled pool.  
> Pipeline vs random: NDCG@10 Δ = +0.237, **p < 0.001** (permutation test, n = 10,000).  
> Rankings are **fully deterministic** — 0 rank swaps across 52 candidates on independent runs.

| Metric | Pipeline | Vector-Only | Random |
|---|---|---|---|
| NDCG@10 | **1.000** [1.000, 1.000] | 0.984 [0.898, 1.000] | 0.763 [0.620, 0.905] |
| Recall@10 (Tier-1) | **1.000** [1.000, 1.000] | 0.900 [0.380, 1.000] | 0.400 [0.030, 0.770] |
| Kendall's τ | **0.825** [0.779, 0.870] | 0.483 [0.282, 0.683] | −0.020 [−0.245, 0.206] |
| Spearman's ρ | **0.935** [0.904, 0.965] | 0.594 [0.358, 0.830] | −0.035 [−0.343, 0.273] |

*Jackknife 95% CIs. Eval methodology: TREC-style qrels, jackknife LOO CI, one-sided permutation test. Full report: [`eval/EVAL.md`](eval/EVAL.md)*

---

## What It Does

1. **Seed a talent pool** — run `seed_candidates.py` once to populate ChromaDB + MongoDB with 45 synthetic candidates across 4 quality tiers (or upload resumes via UI)
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
│              → DeepSeek V3 (OpenRouter) → Resume JSON       │
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
| Model non-determinism | `deepseek/deepseek-chat-v3-0324` — 0 rank swaps across 52 candidates (vs 44 swaps with v3.2) |

Verified: 10 parallel runs → `match_score` range = **3.8 pts** (V3-0324), vs 7.1 pts (Flash). Determinism test: **0/52 rank swaps** across two independent full-pipeline runs.

---

## Evaluation Suite

The `eval/` package provides a production-grade offline evaluation harness:

```bash
# Build ground truth from the 45-candidate labeled seed pool
python eval/run_eval.py build \
  --jd "Senior AI/ML Engineer..." \
  --out eval/qrels.json

# Optionally LLM-judge any uploaded candidates not in the seed set
python eval/run_eval.py build --jd "..." --judge-uploads

# Run full evaluation: pipeline vs vector-only vs random
python eval/run_eval.py evaluate \
  --qrels eval/qrels.json \
  --k 5 10 20
```

**Output:**

```
========================================================================
  PIPELINE EVALUATION  |  45 labeled candidates  |  jd_467db409ed6e628f
========================================================================
  Metric             Pipeline           Vector-Only        Random
  NDCG@10            1.000 [1.000,1.000]   0.984 [0.898,1.000]   0.763 [0.620,0.905]
  Recall@10 (T1)     1.000 [1.000,1.000]   0.900 [0.380,1.000]   0.400 [0.030,0.770]
  Kendall tau        0.845 [0.815,0.876]   0.483 [0.282,0.683]  -0.020 [-0.245,0.206]
  Spearman rho       0.950 [0.931,0.968]   0.594 [0.358,0.830]  -0.035 [-0.343,0.273]
  Pipeline vs random: NDCG@10 delta=+0.237  p<0.001  (permutation n=10,000)
========================================================================
```

Methodology: NDCG@k (primary), jackknife LOO 95% CIs, one-sided permutation test. Full details in [`eval/EVAL.md`](eval/EVAL.md).

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
│                           Candidate ID scheme, filename dedup,
│                           JD hash deduplication, planner plan cache
│
├── reranker.py             Cross-encoder reranker — ms-marco-MiniLM-L-6-v2
│                           Lazy singleton; ce_scores(jd, docs) → 0-1 floats
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
├── eval/                   Offline evaluation suite
│   ├── metrics.py          NDCG, Recall, MAP, MRR, τ, ρ, jackknife CI, permutation test
│   ├── qrels_builder.py    TREC-style qrels from seed tier labels
│   ├── llm_judge.py        LLM-as-judge (V3, 3-run majority vote) for new candidates
│   ├── baselines.py        Random + vector-only baselines
│   ├── run_eval.py         CLI: build / evaluate subcommands
│   ├── qrels.json          Ground truth labels (generated)
│   ├── results/            Per-run JSON metric files
│   └── EVAL.md             Full evaluation report with tables and methodology
│
├── chroma_db/              ChromaDB persistent storage (auto-created)
├── requirements.txt
├── .env                    API keys + DB config
├── .env.example
├── ARCHITECTURE.md         Full technical architecture reference
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

