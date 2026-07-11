# Multi-Agent Resume Screening Assistant — TODO

## Pipeline Architecture

```
PDF/DOCX/TXT  (single or batch upload)
     │
     ▼
ResumeParser (PyPDF2 / python-docx)
     │  extracts raw text
     ▼
OpenRouterClient → DeepSeek V3 (deepseek/deepseek-chat)
     │  LLM structures text into Resume dataclass (JSON)
     ▼
Filename dedup check (MongoDB resumes[] array)
     │  same filename + ≥90% similarity → skip
     │  same filename + <90% similarity → overwrite sub_id
     │  new filename               → new sub_id
     ▼
Candidate ID resolution  (email → base_id + sub_id → chroma_id)
     │  e.g. john.doe@gmail.com → john_doe_2
     ▼
ChromaDB (PersistentClient)  +  MongoDB candidates collection
     │  all-MiniLM-L6-v2 embeddings, cosine similarity
     │  resume_json stored in ChromaDB metadata (no separate lookup)
     ▼
JD hash check → MongoDB jobs (jd_hash match?)
     │
     ├── YES (same JD, new uploads) → Fast path
     │     Evaluate uploaded resumes only (~10-15s)
     │     Load cached pool rankings from MongoDB
     │     Merge + re-rank → sessions for qualifying uploads + top pool
     │
     └── NO → Full LangGraph Screening Pipeline (~90s)
               │
               ├── [1] Planner Node        JD → search queries
               │                           Cached by JD hash (MongoDB jd_plans)
               │                           SCORE_WEIGHTS hardcoded — not LLM-generated
               ├── [2] Retriever Node      Full-pool ChromaDB search (top_k=pool_size)
               ├── [3] Evaluator Node      Parallel LLM scoring (ThreadPoolExecutor, 6 workers)
               │                           Component scores returned; match_score = weighted
               │                           sum in Python (not LLM) for reproducibility
               ├── [4] Critique Node       Read-only QA + flags recommended_for_interview
               │                           No score mutation
               ├── [5] Synthesizer Node    Final ranked list + match_level labels
               └── [6] IQ Generator Node  Parallel tailored questions (top 5, 5 workers)
                         │
                         ▼
               Sessions created for:
                 - Top-5 recommended pool candidates
                 - All uploaded Strong/Good Fit not already covered
     │
     ▼
MongoDB jobs collection  (jd_hash + rankings persisted)
     │
     ▼
══════════════════ LangGraph StateGraph (Interview) ══════════════════
     │
     ├── load_questions          Flatten 5-category question bank
     ├── ask_question            Send Q via WebSocket + TTS audio
     ├── listen                  Receive audio → Whisper/Groq STT
     ├── evaluate                LLM scores answer (1–10)
     ├── decide                  Follow-up or next question
     └── report                  Final verdict, category scores, transcript
══════════════════════════════════════════════════════════════════════
     │
     ▼
MongoDB interview_reports collection  (transcript + verdict persisted)
     │
     ▼
FastAPI + WebSocket UI (interview_app.py)
     Dark-theme single-page app served at http://localhost:8001
```

---

## Features Completed

### Core Pipeline
- [x] Resume parsing — PDF (PyPDF2), DOCX (python-docx), TXT
- [x] LLM-based structured extraction — all fields to JSON
      (contact, skills, experience, education, certifications,
       projects, languages, publications, awards)
- [x] Python dataclass schema — Python 3.14 compatible (no Pydantic)
- [x] ChromaDB vector store — persistent, cosine similarity,
      all-MiniLM-L6-v2 embeddings
- [x] `resume_json` stored in ChromaDB metadata — evaluator works
      without in-memory candidate cache across server restarts
- [x] LangGraph StateGraph (screening) — 6-node typed pipeline
- [x] Planner agent — decomposes JD into search strategy
- [x] Retriever agent — full-pool semantic search (top_k=pool_size),
      no artificial cap, ensures all candidates evaluated
- [x] Evaluator agent — parallel LLM scoring (ThreadPoolExecutor, 6 workers)
- [x] Critique agent — read-only QA, flags recommended candidates,
      no score mutation
- [x] Synthesizer — final ranked list + match_level labels
      (Strong Fit / Good Fit / Potential Fit / Not Recommended)
- [x] Interview question generator — parallel (5 workers), capped at
      top 5 recommended, 5 categories per candidate
- [x] LangGraph StateGraph (interview simulator) — 7-node loop
- [x] Follow-up injection — shallow answers trigger follow-up Q
      spliced into queue dynamically

### Determinism & Score Stability
- [x] `SCORE_WEIGHTS` hardcoded as module-level constants — not LLM-generated
      `{skills: 0.40, experience: 0.35, education: 0.15, projects: 0.10}`
- [x] `match_score` computed in Python from component scores × fixed weights
      (removed free-form holistic LLM scoring)
- [x] Planner output cached by JD hash in MongoDB `jd_plans` collection
- [x] Critique read-only — removed score adjustment/mutation loop
- [x] Evaluations sorted by `candidate_id` before critique for deterministic ordering
- [x] All LLM calls use DeepSeek V3-0324 (`deepseek/deepseek-chat-v3-0324`), temperature=0.1
      Verified: 10-run stability test → match_score range = 3.8 pts (V3) vs 7.1 pts (Flash)
      Determinism verified: v3-0324 → 0 rank swaps across 52 candidates; v3.2 → 44 swaps

### Voice & UI
- [x] TTS — edge-tts (Microsoft neural, en-US-GuyNeural, free)
- [x] STT (local) — faster-whisper large-v3, VAD filter
- [x] STT (cloud) — Groq Whisper large-v3 (~20x faster than CPU)
- [x] Browser audio — MediaRecorder (WebM) → AudioContext WAV
- [x] FastAPI web UI — dark-theme single-page HTML/JS at port 8001
- [x] Multiple resume uploads — `List[UploadFile]`, loop parses all

### Persistence & Deduplication
- [x] Talent pool seeding — seed_candidates.py, 45 synthetic candidates
      across 4 tiers seeded into ChromaDB + MongoDB
- [x] MongoDB persistence — pymongo, 4 collections:
      candidates / jobs / interview_reports / jd_plans
- [x] Candidate ID scheme — email as primary key, base_id from email
      local-part slug, chroma_id = base_id + "_" + sub_id (integer),
      multiple resume versions accumulate in resumes[] array
- [x] Filename-based resume deduplication:
      same filename + ≥90% Jaccard similarity → skip (reuse sub_id)
      same filename + <90% similarity → overwrite in-place (updated resume)
      new filename → new sub_id
- [x] JD deduplication (fast path) — SHA-256 hash of JD stored on
      every job doc; same JD resubmitted → load cached pool rankings,
      evaluate only uploaded resumes, merge + re-rank (~10-15s)
- [x] Pool-changed warning — banner shown if pool size differs from
      cached run (new candidates added since last screen)

### UI & Sessions
- [x] Auto-interview for uploaded candidates — Strong/Good Fit uploads
      always get IQ generation + Interview button regardless of pool rank
- [x] Rankings UI — full talent pool table with rank, score, match level,
      strengths, Interview button; "Your Uploaded Candidates" highlight
      card at top; past runs history
- [x] Past runs — GET /api/jobs loads previous screening results
      read-only with full rankings (no live sessions)
- [x] Interview reports — GET /api/reports lists completed interviews

### Evaluation Suite
- [x] TREC-style qrels ground truth — tier labels from seed_candidates.SEEDS
      stored as `eval/qrels.json` with grade mapping (tier-1→4, tier-2→3, …)
- [x] IR metrics — NDCG@k, Recall@k (tier-1), MAP@k, MRR, Kendall's τ, Spearman's ρ
- [x] Jackknife leave-one-out 95% CI — correct single-query CI methodology
- [x] One-sided permutation test (n=10,000) — statistical significance vs random
- [x] Three-system comparison: full pipeline vs vector-only vs random baseline
- [x] LLM-as-judge (DeepSeek V3, 3-run majority vote) — labels uploaded
      candidates not in seed set; agreement rate as calibration signal
- [x] CLI harness — `python eval/run_eval.py build / evaluate`
- [x] Results: Kendall's τ = 0.845 [0.815, 0.876], NDCG@10 = 1.000 vs
      random p < 0.001 on 45-candidate labeled pool


### Storage & Persistence
- [x] Persist sessions to MongoDB (`sessions` collection, upsert on create,
      partial update after every answer, `mark_session_completed` on finish)
- [x] Session restoration on server restart — `ws_interview` loads from
      MongoDB if session_id not in memory
- [x] `GET /api/report/{session_id}` — full interview report JSON
- [x] `GET /api/report/{session_id}/pdf` — downloadable PDF (fpdf2)
      Colour-coded verdict badge, score table, category bars, strengths/concerns,
      full Q&A transcript with per-answer scores and feedback


### UI Enhancements (Priority: Medium)
- [x] Interview report download button (PDF or JSON)
      PDF: GET /api/report/{id}/pdf — colour-coded verdict, score table, full transcript
      JSON: UI download button → raw JSON blob
- [x] Progress indicator for estimated transcription time
      4px blue bar (#stt-progress-bar), auto-hides on completion
- [x] Allow restarting interview without re-running screening
      POST /api/session/{id}/restart — resets state, keeps IQs; button in UI
- [x] Show raw Whisper text alongside evaluated text
      Hide/Show toggle (#raw-transcript); AI evaluation notes in blue left-border box

### Ranking Improvements (Priority: Medium)
- [x] Cross-encoder reranker after ChromaDB retrieval
      reranker.py — ms-marco-MiniLM-L-6-v2, lazy singleton, sigmoid-normalised scores
- [x] Score fusion: semantic similarity + LLM eval score (RRF)
      Weighted RRF: LLM 50% + cross-encoder 30% + vector 20%
      rrf_fuse() in agentic_pipeline.py; fused_score/ce_score/vector_similarity persisted to MongoDB

---

## What's Next

### STT Quality (Priority: High)
- [ ] Test Groq STT accuracy on Indian English accent vs local large-v3
- [x] Add raw vs cleaned transcription toggle in UI for transparency
      Raw Whisper text shown in #raw-transcript; Hide/Show toggle button
- [x] Auto-fallback from Groq to local faster-whisper on 403/401 errors
      Groq daily limit or key rejection → seamless fallback, status sent via WebSocket



### Fast-Path Optimisations (Priority: Medium)
- [x] Pre-dedup: skip LLM parse for re-uploaded resumes
      Raw text extraction → regex email → MongoDB filename lookup → Jaccard ≥ 0.85 → skip
- [x] Lazy IQ generation for pool candidates
      POST /api/prepare-interview generates on demand; uploaded Strong/Good Fit get upfront parallel IQ

### Evaluation (Priority: Low)
- [ ] Multi-JD evaluation — run eval across 3-5 different JDs, macro-average metrics
      (currently 1 JD / 45 labeled candidates in qrels.json)
- [ ] Calibration check — for LLM-as-judge: % of judgments a human would agree with
- [ ] Eval regression gate — manual `python eval/run_eval.py evaluate` currently;
      no CI pipeline in this repo

### Scaling (Priority: Low)
- [ ] Batch resume ingestion endpoint (ZIP upload)
- [ ] Configurable pool tier filtering in UI
