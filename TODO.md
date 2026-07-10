# Multi-Agent Resume Screening Assistant — TODO

## Pipeline Architecture

```
PDF/DOCX/TXT
     │
     ▼
ResumeParser (PyPDF2)
     │  extracts raw text
     ▼
OpenRouterClient → DeepSeek v4 Flash
     │  LLM structures text into Resume JSON
     ▼
ChromaDB (PersistentClient)
     │  all-MiniLM-L6-v2 embeddings, cosine similarity
     ▼
══════════════════ LangGraph StateGraph ══════════════════
     │
     ├── [1] Planner Node
     │       Decomposes JD into search queries, scoring
     │       weights, hard requirements, nice-to-haves
     │
     ├── [2] Retriever Node
     │       Runs each search query against ChromaDB
     │       Deduplicates + reranks by cosine similarity
     │
     ├── [3] Evaluator Node
     │       Scores each candidate vs JD
     │       Outputs: match_score, strengths, weaknesses,
     │       matched/missing skills, recommendation
     │
     ├── [4] Critique Node
     │       Validates scores for consistency
     │       Applies adjustments, flags recommended candidates
     │
     ├── [5] Synthesizer Node
     │       Final ranked list sorted by match_score
     │       Marks recommended_for_interview = true/false
     │
     └── [6] Interview Question Generator Node
             Fires only for recommended candidates
             Generates per-candidate tailored questions:
             - Technical depth (with follow-ups)
             - Gap probing (tied to specific weaknesses)
             - Project-specific (from their actual resume)
             - System design (role-relevant)
             - Behavioral (grounded in real gaps)
══════════════════════════════════════════════════════════
     │
     ▼
FastAPI Server (main.py)
     REST endpoints for upload, parse, screen, evaluate
```

---

## Features Completed

- [x] Resume parsing — PDF (PyPDF2), DOCX (python-docx), TXT
- [x] LLM-based structured extraction — all fields to JSON
      (contact, skills, experience, education, certifications,
       projects, languages, publications, awards)
- [x] ChromaDB vector store — persistent, cosine similarity,
      all-MiniLM-L6-v2 embeddings
- [x] LangGraph StateGraph — 6-node typed pipeline with
      Annotated reducers for list accumulation
- [x] Planner agent — decomposes JD into search strategy
- [x] Retriever agent — semantic search, dedup, rerank
- [x] Evaluator agent — per-candidate LLM scoring vs JD
- [x] Critique agent — consistency validation + score adjustment
- [x] Synthesizer — final ranked list
- [x] Interview question generator — 5 categories, tailored
      per candidate based on their actual resume + gaps
- [x] FastAPI server — upload, parse, screen, evaluate endpoints
- [x] Bulk resume upload endpoint
- [x] OpenRouter integration — DeepSeek v4 Flash

---

## What's Next

### Interview Simulator (Priority: High)
- [ ] LangGraph loop: ask → answer → evaluate → follow-up or next
- [ ] CLI interface: interactive Q&A session in terminal
- [ ] Answer evaluator: scores each response against expected signal
- [ ] Follow-up decision: if answer is shallow, probe deeper
- [ ] Final interview report: per-question scores + overall verdict

### FastAPI Enhancements (Priority: Medium)
- [ ] Wire all pipeline stages to REST endpoints
- [ ] `POST /interview/start/{candidate_id}` — kick off session
- [ ] `POST /interview/answer` — submit answer, get next question
- [ ] `GET /interview/report/{session_id}` — final assessment
- [ ] WebSocket support for real-time interview streaming

### Storage & Persistence (Priority: Medium)
- [ ] Persist candidate store across server restarts (SQLite/JSON)
- [ ] Session tracking for multi-candidate interview runs
- [ ] Store screening + interview results to disk

### Reranking (Priority: Low)
- [ ] Add cross-encoder reranker after ChromaDB retrieval
      (e.g. ms-marco-MiniLM reranker via sentence-transformers)
- [ ] Score fusion: combine semantic similarity + LLM eval score

### Output (Priority: Low)
- [ ] Candidate PDF report (screening summary for hiring manager)
- [ ] Export interview questions to structured JSON/PDF per candidate
