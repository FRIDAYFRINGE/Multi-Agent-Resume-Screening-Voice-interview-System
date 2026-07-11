# Multi-Agent Resume Screening Assistant — TODO

## Pipeline Architecture

```
PDF/DOCX/TXT
     │
     ▼
ResumeParser (PyPDF2 / python-docx)
     │  extracts raw text
     ▼
OpenRouterClient → DeepSeek v4 Flash
     │  LLM structures text into Resume dataclass (JSON)
     ▼
ChromaDB (PersistentClient)
     │  all-MiniLM-L6-v2 embeddings, cosine similarity
     ▼
══════════════════ LangGraph StateGraph (Screening) ══════════════════
     │
     ├── [1] Planner Node        JD → search queries + scoring weights
     ├── [2] Retriever Node      ChromaDB semantic search + dedup/rerank
     ├── [3] Evaluator Node      LLM scores each candidate vs JD
     ├── [4] Critique Node       Validates scores, flags recommended
     ├── [5] Synthesizer Node    Final ranked list
     └── [6] IQ Generator Node   Per-candidate tailored interview questions
══════════════════════════════════════════════════════════════════════
     │
     ▼
══════════════════ LangGraph StateGraph (Interview) ══════════════════
     │
     ├── load_questions          Flatten 5-category question bank
     ├── ask_question            Send Q via WebSocket + TTS audio
     ├── listen                  Receive audio → Whisper STT
     ├── evaluate                LLM scores answer (1–10)
     ├── decide                  Follow-up or next question
     └── report                  Final verdict, category scores, transcript
══════════════════════════════════════════════════════════════════════
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
- [x] LangGraph StateGraph (screening) — 6-node typed pipeline with
      Annotated reducers for list accumulation
- [x] Planner agent — decomposes JD into search strategy
- [x] Retriever agent — semantic search, dedup, rerank
- [x] Evaluator agent — per-candidate LLM scoring vs JD
- [x] Critique agent — consistency validation + score adjustment
- [x] Synthesizer — final ranked list
- [x] Interview question generator — 5 categories, tailored
      per candidate (technical depth, gap probing, project-specific,
      system design, behavioral)
- [x] LangGraph StateGraph (interview simulator) — 7-node loop:
      load → ask → listen → evaluate → decide → follow-up → report
- [x] Follow-up injection — shallow answers trigger follow-up Q
      spliced into queue dynamically
- [x] TTS — edge-tts (Microsoft neural, en-US-GuyNeural, free)
      streamed to browser as base64 MP3 via WebSocket
- [x] STT (local) — faster-whisper large-v3, VAD filter,
      domain initial_prompt, condition_on_previous_text=False
- [x] STT (cloud) — Groq Whisper large-v3 API (~20x faster than CPU)
      switchable via STT_PROVIDER = "local" | "groq" in interview_app.py
- [x] Browser audio — MediaRecorder (WebM) → AudioContext WAV
      conversion client-side, no ffmpeg dependency
- [x] FastAPI web UI — dark-theme single-page HTML/JS
      served at GET /, WebSocket /ws/{session_id}
- [x] POST /api/screen — upload resume + JD → full pipeline → session
- [x] POST /api/submit-answer — audio upload → STT → LLM eval → WS push
- [x] Silence guard — sub-4-word answers score 0 without LLM call
- [x] Final report — verdict (HIRE/STRONG_HIRE/HOLD/REJECT),
      category scores, strengths, concerns, full transcript
- [x] UI/backend separation — HTML/CSS/JS in ui.py, logic in interview_app.py
- [x] OpenRouter integration — DeepSeek v4 Flash

---

## What's Next

### STT Quality (Priority: High)
- [ ] Test Groq STT accuracy vs local large-v3 on Indian English accent
- [ ] Groq: validate API key whitespace handling in .env
- [ ] Consider adding display of raw vs cleaned transcription toggle in UI

### Storage & Persistence (Priority: Medium)
- [ ] Persist sessions to disk (currently in-memory, lost on server restart)
- [ ] Store final interview reports as JSON to ./reports/
- [ ] Add GET /report/{session_id} download endpoint
- [ ] SQLite for candidate + session tracking across restarts

### UI Enhancements (Priority: Medium)
- [ ] Interview report download button (PDF or JSON)
- [ ] Progress indicator showing estimated transcription time
- [ ] Allow restarting interview without re-running screening
- [ ] Show raw Whisper text alongside evaluated text for transparency

### Reranking (Priority: Low)
- [ ] Add cross-encoder reranker after ChromaDB retrieval
      (ms-marco-MiniLM via sentence-transformers)
- [ ] Score fusion: semantic similarity + LLM eval score

### Scaling (Priority: Low)
- [ ] Parallel evaluator — asyncio.gather() for N candidate LLM calls
- [ ] Batch resume ingestion endpoint
- [ ] Candidate PDF report export for hiring manager
