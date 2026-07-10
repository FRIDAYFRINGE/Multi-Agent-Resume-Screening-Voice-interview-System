# Agentic Pipeline — Full Architecture

## Overview

Multi-agent RAG system for resume screening. Accepts raw resume files, structures
them into JSON, embeds them into ChromaDB, then runs a LangGraph StateGraph where
specialized agents plan, retrieve, evaluate, critique, rank, and generate tailored
interview questions — all driven by a job description.

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INGESTION LAYER                                │
│                                                                         │
│   resume.pdf / .docx / .txt                                             │
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
│   │   client.py)         │  Prompt: structured extraction prompt        │
│   │                      │  → JSON string                               │
│   │  extract_resume_     │                                              │
│   │  from_text()         │  _dict_to_resume() maps raw dict →          │
│   └──────────┬───────────┘  nested dataclass hierarchy                 │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  Resume (dataclass)   schemas.py                         │         │
│   │                                                          │         │
│   │  contact_info: ContactInfo                               │         │
│   │    name, email, phone, address, city, state, country     │         │
│   │    zip_code, linkedin, github, portfolio, website        │         │
│   │                                                          │         │
│   │  professional_summary: str                               │         │
│   │                                                          │         │
│   │  skills: List[Skill]                                     │         │
│   │    category, skills[]                                    │         │
│   │                                                          │         │
│   │  experience: List[Experience]                            │         │
│   │    company, position, start_date, end_date,              │         │
│   │    employment_type, location, description,               │         │
│   │    achievements[], technologies[]                        │         │
│   │                                                          │         │
│   │  education: List[Education]                              │         │
│   │    institution, degree, field_of_study,                  │         │
│   │    start_date, end_date, gpa, activities                 │         │
│   │                                                          │         │
│   │  certifications: List[Certification]                     │         │
│   │    title, issuer, date_obtained, expiration, url         │         │
│   │                                                          │         │
│   │  projects: List[Project]                                 │         │
│   │    title, description, technologies[], url               │         │
│   │                                                          │         │
│   │  languages, publications, volunteer_experience,          │         │
│   │  awards_recognition, metadata                            │         │
│   └──────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Resume dataclass
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
│   │  _resume_to_text() flattens Resume → single string:      │         │
│   │    summary + all skills + positions + descriptions +     │         │
│   │    achievements + technologies + degrees + cert titles   │         │
│   │    + project descriptions                                │         │
│   │                                                          │         │
│   │  Stored per document:                                    │         │
│   │    id       → candidate_id (str)                         │         │
│   │    document → flattened resume text  (embedded)          │         │
│   │    metadata → {name, email, skills_flat}                 │         │
│   └──────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ candidates indexed
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
│  │  min_experience_years   Optional[int]            │                  │
│  │  required_education     Optional[str]            │                  │
│  │  evaluation_criteria    Optional[dict]           │                  │
│  │  plan                   Optional[dict]      ──── │── written by [1] │
│  │  search_queries         List[str]           ──── │── written by [1] │
│  │  retrieved_ids          List[str]           ──── │── written by [2] │
│  │  evaluations            List[dict]  (+add)  ──── │── written by [3] │
│  │  critique               Optional[dict]      ──── │── written by [4] │
│  │  final_rankings         List[dict]          ──── │── written by [5] │
│  │  interview_questions    Dict[str,Any]       ──── │── written by [6] │
│  │  errors                 List[str]   (+add)  ──── │── any node       │
│  └──────────────────────────────────────────────────┘                  │
│                                                                         │
│  Note: evaluations and errors use Annotated[List, operator.add]        │
│  so multiple nodes can append without overwriting each other            │
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
│  Agent   : LLM (DeepSeek v4 Flash)                              │
│                                                                  │
│  INPUT  (reads from state)                                       │
│    job_description, required_skills,                             │
│    min_experience_years, required_education                      │
│                                                                  │
│  PROMPT STRATEGY                                                 │
│    Asks LLM to act as "recruiting planner"                       │
│    temp=0.2 for deterministic output                             │
│                                                                  │
│  OUTPUT (writes to state)                                        │
│    plan: {                                                       │
│      search_queries        List[str]  (3 semantic queries)       │
│      evaluation_focus_areas List[str]                            │
│      scoring_weights        {skills, experience,                 │
│                              education, projects}  (sum=1.0)    │
│      hard_requirements      List[str]  (must-haves)             │
│      nice_to_have           List[str]  (optional)               │
│      top_candidate_count    int                                  │
│    }                                                             │
│    search_queries: List[str]  (copied from plan for next node)  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [2] RETRIEVER NODE                                              │
│                                                                  │
│  Role    : Semantic search over ChromaDB to find relevant        │
│            candidates for each search query                      │
│  Agent   : ChromaDB (no LLM call)                               │
│                                                                  │
│  INPUT  (reads from state)                                       │
│    search_queries  List[str]                                     │
│                                                                  │
│  PROCESS                                                         │
│    For each query → collection.query(query_texts, n_results=10)  │
│    Embedding: all-MiniLM-L6-v2 (384-dim, ONNX)                  │
│    Distance:  cosine  →  similarity = 1 - distance              │
│    Deduplication: set() across all query results                 │
│    Rerank: sort by similarity_score descending                   │
│    Cap at top 20 candidates                                      │
│                                                                  │
│  OUTPUT (writes to state)                                        │
│    retrieved_ids: List[str]   (candidate IDs, ranked)           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [3] EVALUATOR NODE                                              │
│                                                                  │
│  Role    : Score every retrieved candidate against the JD        │
│  Agent   : LLM per candidate (parallel-ready, currently serial) │
│                                                                  │
│  INPUT  (reads from state)                                       │
│    retrieved_ids, job_description, plan                          │
│    candidate_store (in-memory Resume objects)                    │
│                                                                  │
│  PROMPT STRATEGY                                                 │
│    Passes full Resume JSON + JD + scoring_weights                │
│    + hard_requirements + nice_to_have                            │
│    temp=0.2, one LLM call per candidate                         │
│                                                                  │
│  OUTPUT per candidate (appended to state.evaluations)           │
│    {                                                             │
│      candidate_id          str                                   │
│      candidate_name        str                                   │
│      match_score           float  0–100                          │
│      skills_score          float  0–100                          │
│      experience_score      float  0–100                          │
│      education_score       float  0–100                          │
│      strengths             List[str]                             │
│      weaknesses            List[str]                             │
│      matched_skills        List[str]                             │
│      missing_skills        List[str]                             │
│      meets_hard_requirements bool                                │
│      overall_recommendation STRONG_MATCH|MATCH|                 │
│                             WEAK_MATCH|NOT_QUALIFIED             │
│      reasoning             str                                   │
│      suggested_interview_questions List[str]                     │
│    }                                                             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [4] CRITIQUE NODE                                               │
│                                                                  │
│  Role    : QA reviewer — validates all evaluations for           │
│            consistency, catches score inflation/deflation,       │
│            applies corrections, decides who gets interviewed     │
│  Agent   : LLM (single call over all evaluations)               │
│                                                                  │
│  INPUT  (reads from state)                                       │
│    evaluations, job_description                                  │
│                                                                  │
│  PROMPT STRATEGY                                                 │
│    Checks: score inflation, ignored hard requirements,           │
│    inconsistent recommendations between candidates               │
│    temp=0.2                                                      │
│                                                                  │
│  POST-PROCESSING                                                 │
│    Applies adjustments dict back into evaluations in-place       │
│    Marks ev["score_adjusted"] = True on changed scores          │
│                                                                  │
│  OUTPUT (writes to state)                                        │
│    critique: {                                                   │
│      is_valid                bool                                │
│      issues_found            List[str]                           │
│      adjustments             [{candidate_id, adjusted_score,     │
│                                reason}]                          │
│      recommended_for_interview List[str]  (candidate IDs)       │
│      overall_quality         GOOD|ACCEPTABLE|NEEDS_REVIEW        │
│      summary                 str                                 │
│    }                                                             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [5] SYNTHESIZER NODE                                            │
│                                                                  │
│  Role    : Produce the final ranked list                         │
│  Agent   : Deterministic (no LLM call)                          │
│                                                                  │
│  INPUT  (reads from state)                                       │
│    evaluations, critique.recommended_for_interview               │
│                                                                  │
│  PROCESS                                                         │
│    Sort evaluations by match_score descending                    │
│    Assign rank = 1, 2, 3 ...                                     │
│    Set recommended_for_interview = True if ID in critique list  │
│                                                                  │
│  OUTPUT (writes to state)                                        │
│    final_rankings: List[{                                        │
│      rank                    int                                 │
│      candidate_id            str                                 │
│      candidate_name          str                                 │
│      match_score             float                               │
│      overall_recommendation  str                                 │
│      recommended_for_interview bool                              │
│      strengths, weaknesses, matched_skills, missing_skills       │
│      reasoning, score_adjusted (if critique changed it)          │
│    }]                                                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  [6] INTERVIEW QUESTION GENERATOR NODE                           │
│                                                                  │
│  Role    : For each recommended candidate, generate a full       │
│            tailored interview question bank                      │
│  Agent   : LLM per recommended candidate                        │
│  Gate    : Only fires for recommended_for_interview = True       │
│                                                                  │
│  INPUT  (reads from state)                                       │
│    final_rankings (filtered to recommended only)                 │
│    candidate_store (Resume objects)                              │
│    job_description                                               │
│    per-candidate: match_score, strengths, weaknesses,           │
│                   missing_skills, matched_skills                 │
│                                                                  │
│  PROMPT STRATEGY                                                 │
│    Passes full Resume JSON + JD + evaluation breakdown           │
│    Instructs LLM to generate questions that are:                 │
│    - Non-generic (tied to their actual resume content)           │
│    - Gap-targeted (each weakness → a question)                   │
│    - Project-specific (named projects from resume)              │
│    temp=0.4, max_tokens=2048                                     │
│                                                                  │
│  OUTPUT per candidate (writes to state.interview_questions)      │
│    {candidate_id}: {                                             │
│      candidate_name   str                                        │
│      rank             int                                        │
│      match_score      float                                      │
│      technical_depth  [{question, what_to_look_for, follow_up}] │
│      gap_probing      [{gap, question, what_to_look_for}]       │
│      project_specific [{project, question, what_to_look_for}]   │
│      system_design    [{question, what_to_look_for}]            │
│      behavioral       [{question, what_to_look_for}]            │
│    }                                                             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                      END
```

---

## Graph Edge Definition

```python
START → planner → retriever → evaluator → critique → synthesizer → interview_questions → END
```

All edges are deterministic (no conditional branching currently).
State flows through every node in sequence.

---

## LLM Call Distribution

```
Node                    LLM Calls       When
────────────────────────────────────────────────────────────
Ingestion (parser)      1 per resume    On upload / add_candidate()
Planner                 1               Once per screen_candidates()
Retriever               0               ChromaDB only
Evaluator               N               1 per retrieved candidate
Critique                1               Once, over all evaluations
Synthesizer             0               Deterministic sort
Interview Questions     M               1 per recommended candidate
────────────────────────────────────────────────────────────
Total per screening     1 + N + 1 + M  where N = retrieved, M <= N
```

---

## Data Flow Summary

```
ScreeningQuery (JD + filters)
        │
        ├──[Planner]──────► plan + search_queries
        │
        ├──[Retriever]────► retrieved_ids   (ChromaDB cosine search)
        │
        ├──[Evaluator]────► evaluations[]   (1 LLM call per candidate)
        │
        ├──[Critique]─────► critique        (adjusts scores in-place)
        │
        ├──[Synthesizer]──► final_rankings  (sorted, ranked, flagged)
        │
        └──[IQ Generator]► interview_questions{}  (per recommended)
```

---

## File Structure

```
interview_agentic/
│
├── schemas.py              Resume dataclass hierarchy (ContactInfo,
│                           Experience, Education, Skill, Project,
│                           Certification, Language, Resume)
│
├── resume_parser.py        File → raw text extraction
│                           PDF via PyPDF2, DOCX via python-docx
│
├── openrouter_client.py    OpenRouter API wrapper
│                           call_llm(), extract_resume_from_text(),
│                           evaluate_candidate(), _dict_to_resume()
│
├── agentic_pipeline.py     Core pipeline
│                           PipelineState (TypedDict)
│                           CandidateVectorStore (ChromaDB)
│                           AgentNodes (all 6 node methods)
│                           build_screening_graph() → compiled graph
│                           ScreeningPipeline (facade)
│
├── main.py                 FastAPI server
│                           POST /upload-resume
│                           POST /parse-resume-text
│                           POST /bulk-upload
│                           POST /screen-candidates
│                           POST /evaluate-candidate/{id}
│                           GET  /candidates
│                           GET  /candidate/{id}
│
├── example_usage.py        CLI demo script
│
├── chroma_db/              ChromaDB persistent storage (auto-created)
│
├── requirements.txt        fastapi, uvicorn, langgraph, chromadb,
│                           PyPDF2, python-docx, python-dotenv, requests
│
├── .env                    OPENROUTER_API_KEY, OPENROUTER_MODEL
├── TODO.md                 Feature backlog
└── ARCHITECTURE.md         This file
```

---

## Cost Analysis

### Token Length Assumptions

```
Document              Avg Words    Avg Tokens    Notes
──────────────────────────────────────────────────────────────────────
Resume (raw text)     400–600 w    500–800 tok   1 page = ~400w,
                                                 2 page = ~700w
                      assumed avg: 600 tokens raw text

Extraction prompt     —            ~300 tok      Fixed system prompt
                                                 + JSON schema

Resume JSON output    —            ~800 tok      Structured fields,
                                                 all sections filled

Job Description (JD)  200–400 w    ~500 tok      Typical JD with
                                                 requirements, role,
                                                 responsibilities

Evaluation prompt     —            ~300 tok      Fixed recruiter prompt
                                                 + weights + hard reqs

Evaluation output     —            ~500 tok      JSON: scores, skills,
                                                 reasoning, recommendation

Critique input        —            500 tok/eval  JD + N evaluations
                      (per eval)                 stacked in one prompt

IQ prompt             —            ~500 tok      Fixed interviewer prompt
                                                 + strengths/gaps context

IQ output             —            ~1500 tok     5 categories × 3 qs
                                                 with signals + follow-ups
```

---

### Per-Stage Token Breakdown (1 JD × 1000 Resumes)

```
Stage                   Calls    Input Tokens        Output Tokens
────────────────────────────────────────────────────────────────────────
Ingestion               1000     (600 raw + 300       800 (JSON output)
(parse resumes)                   prompt) × 1000      per resume
                                 = 900K input         = 800K output

Planner                    1     JD(500) + instr      plan JSON
                                 = ~500 input         = ~300 output

Retriever (ChromaDB)       0     —                    —          FREE
                                 ONNX embeddings run locally

Evaluator               top 20   JD(500) + resume     eval JSON
(1 call/candidate)               JSON(800) + prompt   = ~500 output
                                 (300) = 1600/each
                                 × 20 = 32K input     = 10K output

Critique                   1     JD(500) + 20 evals   critique JSON
                                 (500 each = 10K) +   = ~500 output
                                 prompt(300)
                                 = ~10.8K input

Synthesizer                0     —                    —          FREE
                                 deterministic sort, no LLM

Interview Questions        5     JD(500) + resume      questions JSON
(top 5 recommended)              JSON(800) + eval      = ~1500/each
                                 (600) + prompt(500)   × 5 = 7.5K output
                                 = 2400/each
                                 × 5 = 12K input
────────────────────────────────────────────────────────────────────────
TOTAL                   1026     ~1,155K input        ~818K output
```

---

### Cost Estimate — DeepSeek V3 via OpenRouter

```
Model : deepseek/deepseek-chat  (DeepSeek V3 / v4 Flash)
Rates : $0.14 / 1M input tokens   |   $0.28 / 1M output tokens
```

```
Stage                   Input Cost      Output Cost     Stage Total
────────────────────────────────────────────────────────────────────
Ingestion (1000 resumes) $0.126          $0.224          $0.350
Planner                  $0.000          $0.000          $0.000
Retriever                FREE            FREE            $0.000
Evaluator (top 20)       $0.004          $0.003          $0.007
Critique                 $0.002          $0.000          $0.002
Synthesizer              FREE            FREE            $0.000
Interview Qs (top 5)     $0.002          $0.002          $0.004
────────────────────────────────────────────────────────────────────
TOTAL                    $0.134          $0.229          $0.363
```

---

### Scaling Table

```
Scenario                                          Est. Cost
──────────────────────────────────────────────────────────────
1 JD  ×  1,000 resumes  (first parse + screen)    ~$0.39
1 JD  ×  1,000 resumes  (already parsed, new JD)  ~$0.01
10 JDs ×  1,000 resumes (parse once, 10 screens)  ~$0.49
1 JD  × 10,000 resumes  (first parse + screen)    ~$3.64
1 JD  × 10,000 resumes  (already parsed, new JD)  ~$0.07
```

**Key observations:**

- Ingestion is ~97% of total cost — and it is one-time per resume
- Once 1000 resumes are parsed and indexed in ChromaDB, any new JD
  costs only ~$0.01 to screen against them (Planner + Evaluator +
  Critique + Interview Qs)
- ChromaDB embeddings (all-MiniLM-L6-v2 via ONNX) run entirely
  locally — zero cost regardless of scale
- Cost per resume to parse: ~$0.00039  (less than 0.04 cents)
- Cost per JD screening run (post-parse): ~$0.013

---

## Technology Stack

```
Layer               Technology              Purpose
─────────────────────────────────────────────────────────────────
File parsing        PyPDF2, python-docx     Extract raw text
LLM                 DeepSeek v4 Flash       Structured extraction,
                    via OpenRouter          evaluation, critique,
                                            interview questions
Embeddings          all-MiniLM-L6-v2        384-dim sentence embeddings
                    (ChromaDB default)      via ONNX runtime
Vector DB           ChromaDB 1.5.x          Persistent cosine search
                    PersistentClient        HNSW index
Orchestration       LangGraph 1.2.x         StateGraph, typed state,
                                            Annotated reducers
Data models         Python dataclasses      Resume schema, no Pydantic
                                            (Python 3.14 compat)
API                 FastAPI 0.115.x         REST endpoints
Runtime             Python 3.14, Windows    Local dev
```
