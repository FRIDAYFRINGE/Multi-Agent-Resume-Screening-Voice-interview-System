# Multi-Agent Resume Screening Assistant — TODO

## Pipeline Architecture

```
PDF/DOCX/TXT  (single or batch upload)
     │
     ▼
ResumeParser (PyPDF2 / python-docx)
     │  extracts raw text
     ▼
OpenRouterClient → DeepSeek v4 Flash
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
               ├── [1] Planner Node        JD → search queries + scoring weights
               ├── [2] Retriever Node      Full-pool ChromaDB search (top_k=pool_size), dedup/rerank
               ├── [3] Evaluator Node      Parallel LLM scoring (ThreadPoolExecutor, 6 workers)
               ├── [4] Critique Node       Validates scores, flags recommended
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
- [x] Critique agent — consistency validation + score adjustment
- [x] Synthesizer — final ranked list + match_level labels
      (Strong Fit / Good Fit / Potential Fit / Not Recommended)
- [x] Interview question generator — parallel (5 workers), capped at
      top 5 recommended, 5 categories per candidate
- [x] LangGraph StateGraph (interview simulator) — 7-node loop
- [x] Follow-up injection — shallow answers trigger follow-up Q
      spliced into queue dynamically
- [x] TTS — edge-tts (Microsoft neural, en-US-GuyNeural, free)
- [x] STT (local) — faster-whisper large-v3, VAD filter
- [x] STT (cloud) — Groq Whisper large-v3 (~20x faster than CPU)
- [x] Browser audio — MediaRecorder (WebM) → AudioContext WAV
- [x] FastAPI web UI — dark-theme single-page HTML/JS at port 8001
- [x] Multiple resume uploads — `List[UploadFile]`, loop parses all
- [x] Talent pool seeding — seed_candidates.py, 49 synthetic candidates
      across 4 tiers (Tier 1–4) seeded into ChromaDB + MongoDB
- [x] MongoDB persistence — pymongo, 3 collections:
      candidates / jobs / interview_reports
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
- [x] Auto-interview for uploaded candidates — Strong/Good Fit uploads
      always get IQ generation + Interview button regardless of pool rank
- [x] Rankings UI — full talent pool table with rank, score, match level,
      strengths, Interview button; "Your Uploaded Candidates" highlight
      card at top; past runs history
- [x] Past runs — GET /api/jobs loads previous screening results
      read-only with full rankings (no live sessions)
- [x] Interview reports — GET /api/reports lists completed interviews
- [x] MongoDB Compass — data visible under `interview_agentic` database

---

## What's Next

### STT Quality (Priority: High)
- [ ] Test Groq STT accuracy on Indian English accent vs local large-v3
- [ ] Add raw vs cleaned transcription toggle in UI for transparency

### Storage & Persistence (Priority: Medium)
- [ ] Persist sessions to MongoDB (currently in-memory, lost on restart)
- [ ] Add GET /report/{session_id} download endpoint
- [ ] Interview report PDF export for hiring manager

### UI Enhancements (Priority: Medium)
- [ ] Interview report download button (PDF or JSON)
- [ ] Progress indicator for estimated transcription time
- [ ] Allow restarting interview without re-running screening
- [ ] Show raw Whisper text alongside evaluated text

### Ranking Improvements (Priority: Medium)
- [ ] Cross-encoder reranker after ChromaDB retrieval
      (ms-marco-MiniLM via sentence-transformers)
- [ ] Score fusion: semantic similarity + LLM eval score (RRF)

### Scaling (Priority: Low)
- [ ] Batch resume ingestion endpoint (ZIP upload)
- [ ] Candidate PDF report export
- [ ] Configurable pool tier filtering in UI
