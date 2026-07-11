# Agentic Pipeline — Full Architecture

## Overview

End-to-end multi-agent system for resume screening and live voice interviews.
Accepts raw resume files (single or batch), structures them into JSON, deduplicates
by filename + content similarity, embeds into ChromaDB, runs a 6-node LangGraph
screening pipeline with full-pool evaluation and parallel LLM calls, persists all
results to MongoDB, then hands off recommended candidates to a 7-node LangGraph
interview simulator with real-time TTS/STT via a FastAPI WebSocket UI.

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INGESTION LAYER                                │
│                                                                         │
│   resume.pdf / .docx / .txt  (one or many via UI)                      │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────┐                                                   │
│   │  ResumeParser   │  PyPDF2 / python-docx / open()                   │
│   │  (resume_       │  → raw text string                               │
│   │   parser.py)    │                                                   │
│   └────────┬────────┘                                                   │
│            │ raw text                                                   │
│            ▼                                                            │
│   ┌──────────────────────┐                                              │
│   │  OpenRouterClient    │  POST /chat/completions                      │
│   │  (openrouter_        │  Model: deepseek/deepseek-v4-flash           │
│   │   client.py)         │  → Resume dataclass (JSON)                  │
│   └──────────┬───────────┘                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  Candidate ID Resolution  (database.py)                  │         │
│   │                                                          │         │
│   │  email → base_id (slug from email local-part)            │         │
│   │  base_id + sub_id → chroma_id  (e.g. john_doe_2)        │         │
│   │  sub_id source: MongoDB latest_sub_id → +1               │         │
│   │  doc_key: email if available, else base_id               │         │
│   │                                                          │         │
│   │  Filename deduplication (before sub_id increment):       │         │
│   │    find_resume_by_filename(doc_key, filename)            │         │
│   │    ┌─ match found ──────────────────────────────┐        │         │
│   │    │  Jaccard similarity ≥ 90%  → SKIP          │        │         │
│   │    │  (same resume re-uploaded, reuse chroma_id) │        │         │
│   │    │  Jaccard similarity < 90%  → OVERWRITE      │        │         │
│   │    │  (updated resume, overwrite same sub_id)    │        │         │
│   │    └─ no match ──────────────────────────────────┘        │         │
│   │       new filename → new sub_id → new chroma_id           │         │
│   └──────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Resume dataclass + chroma_id
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         VECTOR STORE LAYER                              │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  CandidateVectorStore   (agentic_pipeline.py)            │         │
│   │                                                          │         │
│   │  Backend : ChromaDB PersistentClient  ./chroma_db/       │         │
│   │  Model   : all-MiniLM-L6-v2  (384-dim, ONNX runtime)    │         │
│   │  Distance: cosine similarity  (hnsw:space=cosine)        │         │
│   │  Collection: "candidates"                                │         │
│   │                                                          │         │
│   │  Stored per document:                                    │         │
│   │    id        → chroma_id  (e.g. john_doe_2)             │         │
│   │    document  → flattened resume text  (embedded)         │         │
│   │    metadata  → {name, email, skills_flat, resume_json}   │         │
│   │                  ↑ resume_json enables evaluator to work  │         │
│   │                    without in-memory cache on restart     │         │
│   └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  MongoDB  candidates collection  (database.py)           │         │
│   │                                                          │         │
│   │  _id      : email (preferred) or base_id                 │         │
│   │  base_id  : slug (e.g. "john_doe")                       │         │
│   │  latest_sub_id: int  (authoritative version counter)     │         │
│   │  resume_json: dict   (latest version)                    │         │
│   │  resumes[]: [{sub_id, chroma_id, filename, added_at}]   │         │
│   │             one entry per resume version                 │         │
│   └──────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ candidates indexed
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      JD DEDUPLICATION CHECK                             │
│                                                                         │
│  make_jd_hash(jd) → SHA-256[:16] hex string                            │
│  find_job_by_hash(hash) → most recent matching job doc or None         │
│                                                                         │
│  ┌─ Same JD + uploads present ──────────────────────────────────┐      │
│  │  FAST PATH (~10-15s)                                         │      │
│  │  1. Evaluate uploaded candidates only (parallel, ≤5 workers) │      │
│  │  2. Load cached pool rankings from MongoDB jobs collection   │      │
│  │  3. Merge + re-sort by match_score + re-rank                 │      │
│  │  4. Attach match_level labels                                │      │
│  │  5. Create sessions: uploaded Good/Strong Fit + top-5 pool  │      │
│  │  6. Return {rankings, sessions, reused_job_id, reused_date,  │      │
│  │            pool_changed, old_pool_size}                      │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌─ New JD or no uploads ───────────────────────────────────────┐      │
│  │  FULL PIPELINE (below)                                       │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (full pipeline path)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH STATE MACHINE                             │
│                                                                         │
│  Input: ScreeningQuery                                                  │
│    job_description      str                                             │
│    required_skills      List[str]                                       │
│    min_experience_years Optional[int]                                   │
│    required_education   Optional[str]                                   │
│    evaluation_criteria  Optional[dict]                                  │
│                                                                         │
│  PipelineState (TypedDict) — shared across all nodes                   │
│  ┌──────────────────────────────────────────────────┐                  │
│  │  job_description        str                      │                  │
│  │  required_skills        List[str]                │                  │
│  │  plan                   Optional[dict]      ──── │── written by [1] │
│  │  search_queries         List[str]           ──── │── written by [1] │
│  │  retrieved_ids          List[str]           ──── │── written by [2] │
│  │  evaluations            List[dict]  (+add)  ──── │── written by [3] │
│  │  critique               Optional[dict]      ──── │── written by [4] │
│  │  final_rankings         List[dict]          ──── │── written by [5] │
│  │  interview_questions    Dict[str,Any]       ──── │── written by [6] │
│  │  errors                 List[str]   (+add)  ──── │── any node       │
│  └──────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## LangGraph Node Graph

```
  START
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  [1] PLANNER NODE                                                │
│                                                                  │
│  Role    : Decompose the job description into a search strategy  │
│  Agent   : LLM (DeepSeek v4 Flash), temp=0.2                    │
│                                                                  │
│  OUTPUT (writes to state)                                        │
│    plan: {                                                       │
│      search_queries         List[str]  (3 semantic queries)      │
│      evaluation_focus_areas List[str]                            │
│      scoring_weights        {skills, experience,                 │
│                              education, projects}  (sum=1.0)    │
│      hard_requirements      List[str]  (must-haves)             │
│      nice_to_have           List[str]  (optional)               │
│    }                                                             │
│    search_queries: List[str]                                     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [2] RETRIEVER NODE                                              │
│                                                                  │
│  Role    : Retrieve ALL candidates from ChromaDB                 │
│  Agent   : ChromaDB (no LLM call)                               │
│                                                                  │
│  PROCESS                                                         │
│    top_k = pool_size  (full pool — no artificial cap)            │
│    For each query → collection.query(query_texts, n_results=N)   │
│    Deduplication: set() across all query results                 │
│    Rerank: sort by similarity_score descending                   │
│    No [:20] cap — every candidate is evaluated                   │
│                                                                  │
│  OUTPUT                                                          │
│    retrieved_ids: List[str]  (all candidate IDs, ranked)        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [3] EVALUATOR NODE                                              │
│                                                                  │
│  Role    : Score every retrieved candidate against the JD        │
│  Agent   : Parallel LLM calls (ThreadPoolExecutor, max_workers=6)│
│                                                                  │
│  Resume data source (priority order):                            │
│    1. candidate_store (in-memory, if added this request)         │
│    2. ChromaDB metadata resume_json (persistent fallback)        │
│                                                                  │
│  OUTPUT per candidate (appended to state.evaluations)           │
│    {candidate_id, candidate_name, match_score (0-100),          │
│     skills_score, experience_score, education_score,            │
│     strengths[], weaknesses[], matched_skills[], missing_skills[]│
│     meets_hard_requirements, overall_recommendation,            │
│     reasoning}                                                   │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [4] CRITIQUE NODE                                               │
│                                                                  │
│  Role    : QA reviewer — validates all evaluations for           │
│            consistency, catches score inflation/deflation,       │
│            applies corrections, decides who gets interviewed     │
│  Agent   : LLM (single call over all evaluations), temp=0.2     │
│                                                                  │
│  POST-PROCESSING                                                 │
│    Applies score adjustments back into evaluations in-place      │
│    Marks ev["score_adjusted"] = True on changed scores          │
│                                                                  │
│  OUTPUT                                                          │
│    critique: {is_valid, issues_found[], adjustments[],           │
│               recommended_for_interview[], overall_quality,      │
│               summary}                                           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [5] SYNTHESIZER NODE                                            │
│                                                                  │
│  Role    : Produce the final ranked list                         │
│  Agent   : Deterministic (no LLM call)                          │
│                                                                  │
│  PROCESS                                                         │
│    Sort evaluations by match_score descending                    │
│    Assign rank = 1, 2, 3 ...                                     │
│    Set recommended_for_interview = True if ID in critique list  │
│                                                                  │
│  match_level labels added in interview_app.py post-processing:  │
│    match_score ≥ 70  OR STRONG_MATCH  → "Strong Fit"            │
│    match_score ≥ 50  OR MATCH         → "Good Fit"              │
│    match_score ≥ 30  OR WEAK_MATCH    → "Potential Fit"         │
│    otherwise                          → "Not Recommended"        │
│                                                                  │
│  OUTPUT                                                          │
│    final_rankings: List[{rank, candidate_id, candidate_name,     │
│      match_score, match_level, overall_recommendation,           │
│      recommended_for_interview, strengths[], weaknesses[],       │
│      matched_skills[], missing_skills[], reasoning}]             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [6] INTERVIEW QUESTION GENERATOR NODE                           │
│                                                                  │
│  Role    : Generate tailored interview question banks            │
│  Agent   : Parallel LLM (ThreadPoolExecutor, max_workers=5)     │
│  Gate    : recommended_for_interview = True only, capped at top 5│
│                                                                  │
│  PROMPT STRATEGY                                                 │
│    Full Resume JSON + JD + evaluation breakdown                  │
│    Questions are non-generic, gap-targeted, project-specific     │
│    temp=0.4, max_tokens=1400                                     │
│                                                                  │
│  OUTPUT per candidate                                            │
│    {candidate_name, rank, match_score,                           │
│     technical_depth  [{question, what_to_look_for, follow_up}]  │
│     gap_probing      [{gap, question, what_to_look_for}]        │
│     project_specific [{project, question, what_to_look_for}]    │
│     system_design    [{question, what_to_look_for}]             │
│     behavioral       [{question, what_to_look_for}]}            │
│                                                                  │
│  Post-pipeline session creation (interview_app.py):             │
│    Sessions for top-5 recommended pool candidates               │
│    PLUS all uploaded Strong/Good Fit not already covered        │
│    PLUS top-5 pool Strong/Good Fit on fast-path re-runs         │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                      END
```

---

## Graph Edge Definition

```python
START → planner → retriever → evaluator → critique → synthesizer → interview_questions → END
```

All edges are deterministic (no conditional branching).

---

## MongoDB Persistence Layer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONGODB  (database.py)                               │
│                                                                         │
│  URI    : MONGO_URI env var (default: mongodb://localhost:27017)        │
│  DB     : interview_agentic                                             │
│                                                                         │
│  ── candidates collection ─────────────────────────────────────────    │
│  {                                                                      │
│    _id:           email or base_id  (string, not ObjectId)             │
│    base_id:       "john_doe"                                            │
│    name:          "John Doe"                                            │
│    email:         "john.doe@gmail.com"                                  │
│    source:        "uploaded" | "seeded"                                 │
│    latest_sub_id: 2                                                     │
│    resume_json:   {...}   ← always the latest version                  │
│    resumes: [                                                           │
│      {sub_id:1, chroma_id:"john_doe_1", filename:"cv_v1.pdf", ...}    │
│      {sub_id:2, chroma_id:"john_doe_2", filename:"cv_v2.pdf", ...}    │
│    ]                                                                    │
│    added_at, updated_at                                                 │
│  }                                                                      │
│                                                                         │
│  ── jobs collection ───────────────────────────────────────────────    │
│  {                                                                      │
│    _id:                  ObjectId                                       │
│    jd_hash:              "a3f1b2c4..."  ← SHA-256[:16] for dedup      │
│    job_description:      str                                            │
│    required_skills:      []                                             │
│    pool_size:            int                                            │
│    candidates_evaluated: int                                            │
│    rankings:             [{rank, candidate_id, match_score, ...}]      │
│    critique_summary:     str                                            │
│    run_at:               datetime                                       │
│  }                                                                      │
│                                                                         │
│  ── interview_reports collection ──────────────────────────────────    │
│  {                                                                      │
│    _id:             session_id  (UUID string)                           │
│    job_id:          str (ObjectId of the screening run)                │
│    candidate_id:    str                                                 │
│    candidate_name:  str                                                 │
│    job_description: str (first 500 chars)                              │
│    verdict:         HIRE | STRONG_HIRE | HOLD | REJECT                 │
│    overall_score:   float                                               │
│    summary:         str                                                 │
│    transcript:      [{question_idx, category, question, answer,        │
│                        evaluation}]                                     │
│    report:          full report dict                                    │
│    completed_at:    datetime                                            │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Simulator — LangGraph State Machine

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INTERVIEW STATE MACHINE                              │
│                                                                         │
│  Triggered for each candidate with a session                            │
│  (recommended pool candidates + qualifying uploaded candidates)         │
│                                                                         │
│  Question queue flattened from 5 categories:                           │
│    technical_depth → gap_probing → project_specific →                  │
│    system_design → behavioral                                           │
│                                                                         │
│  START                                                                  │
│    │                                                                    │
│    ▼                                                                    │
│  [1] load_questions    Flatten IQ dict → ordered question_queue        │
│    │                                                                    │
│    ▼                                                                    │
│  [2] ask_question      Pop current Q, send via TTS + WebSocket         │
│    │                                                                    │
│    ▼                                                                    │
│  [3] listen            Receive audio → Whisper/Groq STT → text         │
│    │                                                                    │
│    ▼                                                                    │
│  [4] evaluate          LLM scores answer 1–10, hits/misses,            │
│                        needs_follow_up flag                             │
│    │                                                                    │
│    ▼                                                                    │
│  [5] decide                                                             │
│    │  needs_follow_up AND follow_up_count < max_follow_ups?            │
│    │    YES → inject follow_up Q at current_idx+1 → ask_question       │
│    │    NO  → advance current_idx                                       │
│    │  more questions?                                                   │
│    │    YES → ask_question                                              │
│    │    NO  → report                                                    │
│    │                                                                    │
│    ▼                                                                    │
│  [6] report            LLM generates final assessment                  │
│                          overall_score, verdict, category_scores,      │
│                          top_strengths, key_concerns, recommendation   │
│    │                                                                    │
│    ▼                                                                    │
│  MongoDB interview_reports.update_one (upsert by session_id)          │
│   END                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Voice I/O Layer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VOICE I/O  (voice_io.py)                       │
│                                                                         │
│  TTS: edge-tts                                                          │
│    Voice   : en-US-GuyNeural (Microsoft Edge neural)                    │
│    Cost    : Free (no API key)                                          │
│    Output  : MP3 bytes streamed async → base64 → WebSocket              │
│    Latency : ~500ms for a typical question                              │
│                                                                         │
│  STT Local: faster-whisper                                              │
│    Model   : large-v3  (3GB, cached after first download)              │
│    Device  : CPU (int8 quantized)                                       │
│    Speed   : ~0.5–1× real-time on CPU                                  │
│    Options : vad_filter=True, condition_on_previous_text=False,         │
│              initial_prompt = domain vocab + current question           │
│                                                                         │
│  STT Cloud: GroqSTT                                                     │
│    Model   : whisper-large-v3 (same model, cloud GPU)                   │
│    Cost    : Free tier — 28,800s audio/day (~8 hours)                  │
│    Speed   : ~5s for 2min audio (~20× faster than local CPU)           │
│    Key     : GROQ_API_KEY in .env                                       │
│                                                                         │
│  Switch: STT_PROVIDER = "local" | "groq"  (interview_app.py line ~30) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Web UI Layer — FastAPI + WebSocket

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   WEB UI  (interview_app.py + ui.py)                    │
│                                                                         │
│  Server: uvicorn  http://0.0.0.0:8001                                   │
│                                                                         │
│  GET  /                      Dark-theme SPA (served from ui.py)        │
│  GET  /api/pool-count        {count, mongo}                             │
│  GET  /api/jobs              List past screening runs (summary)         │
│  GET  /api/jobs/{id}         Full screening result (read-only)         │
│  GET  /api/reports           List completed interview reports           │
│                                                                         │
│  POST /api/screen  multipart/form-data                                  │
│    Fields: resumes[] (multiple files), jd, skills, max_follow_ups      │
│    Fast path  : same JD detected → cached pool + eval uploads only     │
│    Full path  : 6-node LangGraph pipeline → persist to MongoDB         │
│    Returns: {rankings[], sessions{}, pool_size, uploaded_cids[],       │
│              reused_job_id?, reused_job_date?, pool_changed?,          │
│              old_pool_size?}                                            │
│                                                                         │
│  POST /api/submit-answer  multipart/form-data                           │
│    Fields: audio (WAV), session_id, question_idx                       │
│    Flow: WAV → STT → silence guard → LLM eval → WS push               │
│                                                                         │
│  WS /ws/{session_id}                                                    │
│    Client → Server: {type:"start"}, {type:"next"}                      │
│    Server → Client: {type:"question"}, {type:"tts_audio"},             │
│                     {type:"transcription"}, {type:"evaluation"},        │
│                     {type:"report"}, {type:"status"}                   │
│                                                                         │
│  UI Panels:                                                             │
│    Setup panel      — JD input, skills, file upload, past runs         │
│    Rankings panel   — "Your Uploaded Candidates" highlight card +       │
│                       full talent pool table; Interview buttons for     │
│                       candidates with sessions                          │
│                       Yellow banner on cached-run fast path            │
│    Interview panel  — live Q&A with TTS playback + answer recording    │
│    Report panel     — final verdict, scores, strengths, concerns       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Call Distribution

```
Node / Step              LLM Calls        Notes
────────────────────────────────────────────────────────────────────
Ingestion (parser)       1 per resume     On upload / add_candidate()
Planner                  1                Once per full pipeline run
Retriever                0                ChromaDB only (free)
Evaluator                N (parallel)     1 per candidate, 6 workers
Critique                 1                Once, over all evaluations
Synthesizer              0                Deterministic sort (free)
IQ Generator             ≤5 (parallel)    1 per recommended, 5 workers
Answer evaluator         1 per answer     During live interview
Final report             1 per session    After all questions answered
────────────────────────────────────────────────────────────────────
Fast path (same JD)      U only           U = uploaded candidates
                                          Pool rankings from MongoDB
```

---

## Data Flow Summary

```
ScreeningQuery (JD + filters)
        │
        ├──[JD hash check]──► cached job found?
        │    YES + uploads ──► fast path (eval uploads, merge cached)
        │    NO ──────────────►
        │
        ├──[Planner]──────► plan + search_queries
        │
        ├──[Retriever]────► retrieved_ids   (full pool, ChromaDB cosine)
        │
        ├──[Evaluator]────► evaluations[]   (parallel, 6 workers)
        │
        ├──[Critique]─────► critique        (adjusts scores in-place)
        │
        ├──[Synthesizer]──► final_rankings  (sorted, ranked, labelled)
        │
        ├──[IQ Generator]► interview_questions{}  (parallel, top 5)
        │
        ├──[MongoDB]──────► jobs.insert_one (with jd_hash)
        │
        └──[Sessions]─────► in-memory sessions dict
                            (top-5 pool + all qualifying uploads)
```

---

## File Structure

```
interview_agentic/
│
├── interview_app.py        Main entry point — FastAPI + WebSocket server
│                           STT_PROVIDER = "local" | "groq"
│                           POST /api/screen  (fast path + full pipeline)
│                           POST /api/submit-answer
│                           WebSocket /ws/{session_id}
│                           _match_level(), _generate_iq(),
│                           _evaluate_candidate(), _resume_similarity()
│
├── ui.py                   Full dark-theme single-page frontend
│                           HTML + CSS + JS as HTML string constant
│                           Panels: setup / rankings / interview / report
│                           Rankings: uploaded highlight card + pool table
│                           Cached-run banner, past runs history
│
├── database.py             MongoDB integration (pymongo)
│                           Collections: candidates, jobs, interview_reports
│                           resolve_candidate_id()  — email slug + sub_id
│                           add_resume_version()    — upsert + push to resumes[]
│                           find_resume_by_filename()
│                           update_resume_version_inplace()
│                           make_jd_hash(), find_job_by_hash()
│                           save_job(), get_job(), list_jobs()
│                           save_interview_report(), list_reports()
│
├── agentic_pipeline.py     Screening pipeline
│                           PipelineState (TypedDict)
│                           CandidateVectorStore (ChromaDB)
│                             get_candidate_data() — resume_json from metadata
│                           AgentNodes (all 6 LangGraph nodes)
│                             evaluator_node  — ThreadPoolExecutor(6)
│                             interview_questions_node — ThreadPoolExecutor(5)
│                             retriever_node — full pool (top_k=pool_size)
│                           build_screening_graph() → compiled graph
│                           ScreeningPipeline (facade class)
│
├── interview_simulator.py  LangGraph 7-node interview state machine
│                           _flatten_questions() — 5-category ordering
│                           _parse_json() — robust JSON extraction
│
├── voice_io.py             Voice I/O classes
│                           TextToSpeech  — edge-tts (free, no API key)
│                           SpeechToText  — faster-whisper (local CPU/GPU)
│                           GroqSTT       — Groq cloud Whisper large-v3
│
├── openrouter_client.py    OpenRouter API wrapper
│                           call_llm(), extract_resume_from_text()
│                           Timeout: 60s (handles parallel load)
│
├── resume_parser.py        File → raw text extraction
│                           PDF via PyPDF2, DOCX via python-docx
│
├── schemas.py              Resume dataclass hierarchy
│                           ContactInfo, Experience, Education, Skill
│                           Project, Certification, Language, Resume
│                           Python dataclasses (no Pydantic — Py 3.14)
│
├── seed_candidates.py      Populates ChromaDB + MongoDB with 49 synthetic
│                           candidates across 4 tiers (Tier 1–4)
│                           Run once: python seed_candidates.py
│
├── run_interview.py        CLI entry point (terminal-only, no browser)
├── main.py                 Legacy FastAPI screening-only endpoints
├── example_usage.py        CLI demo / smoke test script
│
├── chroma_db/              ChromaDB persistent storage (auto-created)
│
├── requirements.txt        All dependencies
├── .env                    OPENROUTER_API_KEY, GROQ_API_KEY,
│                           MONGO_URI, MONGO_DB, CHROMA_DB_PATH
├── .env.example            Template for new setups
├── TODO.md                 Feature backlog + pipeline architecture
└── ARCHITECTURE.md         This file
```

---

## Cost Analysis

### Token Length Assumptions

```
Document              Avg Words    Avg Tokens    Notes
──────────────────────────────────────────────────────────────────────
Resume (raw text)     400–600 w    500–800 tok   1 page = ~400w
Extraction prompt     —            ~300 tok      Fixed system prompt
Resume JSON output    —            ~800 tok      Structured fields
Job Description (JD)  200–400 w    ~500 tok
Evaluation prompt     —            ~300 tok      Fixed recruiter prompt
Evaluation output     —            ~500 tok
IQ prompt             —            ~500 tok
IQ output             —            ~1400 tok     capped at max_tokens=1400
```

### Per-Stage Token Breakdown (1 JD × 1000 Resumes, full pipeline)

```
Stage                   Calls    Input Tokens        Output Tokens
────────────────────────────────────────────────────────────────────────
Ingestion               1000     900K input          800K output
Planner                    1     ~500 input          ~300 output
Retriever (ChromaDB)       0     FREE (local ONNX)   —
Evaluator               all N    1600/each           ~500/each
Critique                   1     ~10.8K input        ~500 output
Synthesizer                0     FREE                —
IQ Generator              ≤5     2400/each × 5       1400/each × 5
────────────────────────────────────────────────────────────────────────
Fast path (same JD)     U only   1600/each × U       500/each × U
(U = uploaded resumes)           No planner/retriever/critique/synth
```

### Cost Estimate — DeepSeek V3 via OpenRouter

```
Rates: $0.14 / 1M input tokens  |  $0.28 / 1M output tokens

Scenario                                          Est. Cost
──────────────────────────────────────────────────────────────
1 JD × 1,000 resumes  (first parse + screen)      ~$0.39
1 JD × 1,000 resumes  (already parsed, new JD)    ~$0.01
10 JDs × 1,000 resumes (parse once, 10 screens)   ~$0.49
1 JD × 10,000 resumes  (first parse + screen)     ~$3.64
Same JD re-run + 1 upload (fast path)             ~$0.001
```

**Key observations:**
- Ingestion is ~97% of total cost — and it is one-time per resume
- Fast path re-runs cost near zero (no pipeline, only eval uploaded candidates)
- ChromaDB embeddings run entirely locally — zero cost regardless of scale

---

## Technology Stack

```
Layer               Technology              Purpose
─────────────────────────────────────────────────────────────────
File parsing        PyPDF2, python-docx     Extract raw text
LLM                 DeepSeek v4 Flash       Structured extraction,
                    via OpenRouter          evaluation, critique,
                                            IQ generation, answer
                                            scoring, final reports
Embeddings          all-MiniLM-L6-v2        384-dim sentence embeddings
                    (ChromaDB default)      via ONNX runtime (local)
Vector DB           ChromaDB 1.5.x          Persistent cosine search
                    PersistentClient        HNSW index
Persistence         MongoDB 8.3.x           candidates, jobs,
                    pymongo                 interview_reports
Orchestration       LangGraph 1.2.x         Screening pipeline (6 nodes)
                                            + interview loop (7 nodes)
Data models         Python dataclasses      Resume schema, no Pydantic
                                            (Python 3.14 compat)
Parallelism         ThreadPoolExecutor      Evaluator (6 workers),
                                            IQ generator (5 workers)
TTS                 edge-tts                Microsoft Edge neural voices
                                            Free, no API key, async MP3
STT (local)         faster-whisper          Whisper large-v3 on CPU
                                            VAD filter, initial_prompt
STT (cloud)         Groq Whisper API        Same model on cloud GPU
                                            28,800s/day free, ~20x faster
Audio (browser)     Web Audio API           WebM → WAV via AudioContext
                    MediaRecorder           No ffmpeg dependency
API + WebSocket     FastAPI 0.115.x         REST + real-time interview
                    uvicorn                 WebSocket message passing
Frontend            Vanilla HTML/CSS/JS     Dark-theme SPA, no framework
Runtime             Python 3.14, Windows    Local dev
```
