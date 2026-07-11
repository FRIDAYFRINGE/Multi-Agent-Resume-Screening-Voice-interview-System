import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import operator
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
import chromadb
from chromadb.config import Settings

from openrouter_client import OpenRouterClient
from schemas import Resume


# ─── LangGraph State ──────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    job_description: str
    required_skills: List[str]
    min_experience_years: Optional[int]
    required_education: Optional[str]
    evaluation_criteria: Optional[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    search_queries: List[str]
    retrieved_ids: List[str]
    evaluations: Annotated[List[Dict[str, Any]], operator.add]
    critique: Optional[Dict[str, Any]]
    final_rankings: List[Dict[str, Any]]
    interview_questions: Dict[str, Any]   # {candidate_id: {categories...}}
    errors: Annotated[List[str], operator.add]


@dataclass
class ScreeningQuery:
    job_description: str
    required_skills: List[str] = field(default_factory=list)
    min_experience_years: Optional[int] = None
    required_education: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None


# ─── ChromaDB Vector Store ────────────────────────────────────────────────────

class CandidateVectorStore:
    """ChromaDB-backed vector store for candidate embeddings and semantic search"""

    def __init__(self, persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        # Uses chromadb's default all-MiniLM-L6-v2 embedding function
        self.collection = self.client.get_or_create_collection(
            name="candidates",
            metadata={"hnsw:space": "cosine"}
        )

    def add_candidate(self, resume: Resume, candidate_id: str):
        """Embed and store candidate resume"""
        text = self._resume_to_text(resume)
        name = resume.contact_info.name if resume.contact_info else "Unknown"

        # Upsert so re-adding same ID doesn't error
        self.collection.upsert(
            ids=[candidate_id],
            documents=[text],
            metadatas=[{
                "name": name,
                "email": resume.contact_info.email or "" if resume.contact_info else "",
                "skills_flat": self._skills_flat(resume),
                "resume_json": json.dumps(resume.dict()),
            }]
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search returning top_k candidates"""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count() or 1)
        )

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                candidates.append({
                    "id": cid,
                    "similarity_score": round(1 - distance, 4),  # cosine: distance=0 → similarity=1
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                })
        return candidates

    def count(self) -> int:
        return self.collection.count()

    def get_candidate_data(self, candidate_id: str) -> Optional[dict]:
        """Retrieve resume dict from ChromaDB (stored as JSON in metadata)."""
        try:
            result = self.collection.get(ids=[candidate_id], include=["metadatas"])
            if result and result.get("ids"):
                rj = result["metadatas"][0].get("resume_json", "")
                return json.loads(rj) if rj else None
        except Exception:
            pass
        return None

    def _resume_to_text(self, resume: Resume) -> str:
        parts = []
        if resume.professional_summary:
            parts.append(resume.professional_summary)
        for s in resume.skills:
            parts.extend(s.skills)
        for exp in resume.experience:
            if exp.position:
                parts.append(exp.position)
            if exp.description:
                parts.append(exp.description)
            if exp.technologies:
                parts.extend(exp.technologies)
            if exp.achievements:
                parts.extend(exp.achievements)
        for edu in resume.education:
            if edu.degree:
                parts.append(edu.degree)
            if edu.field_of_study:
                parts.append(edu.field_of_study)
        for cert in resume.certifications:
            if cert.title:
                parts.append(cert.title)
        for proj in resume.projects:
            if proj.description:
                parts.append(proj.description)
            if proj.technologies:
                parts.extend(proj.technologies)
        return " ".join(filter(None, parts))

    def _skills_flat(self, resume: Resume) -> str:
        return ", ".join(skill for s in resume.skills for skill in s.skills)


# ─── Agent Nodes ──────────────────────────────────────────────────────────────

def _parse_json_response(response: str) -> dict:
    """Strip markdown fences and parse JSON"""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


class AgentNodes:
    def __init__(self, llm: OpenRouterClient, store: CandidateVectorStore, candidate_store: Dict[str, Resume]):
        self.llm = llm
        self.store = store
        self.candidates = candidate_store

    # ── Planner ───────────────────────────────────────────────────────────────
    def planner_node(self, state: PipelineState) -> dict:
        """Decompose the job description into a structured screening plan"""
        prompt = f"""You are a recruiting planner. Given the job description below, create a structured screening plan.

Job Description:
{state['job_description']}

Required Skills: {', '.join(state.get('required_skills', []) or [])}
Min Experience: {state.get('min_experience_years')} years
Required Education: {state.get('required_education', 'Not specified')}

Return ONLY valid JSON:
{{
  "search_queries": ["<semantic query 1>", "<semantic query 2>", "<semantic query 3>"],
  "evaluation_focus_areas": ["<area1>", "<area2>"],
  "scoring_weights": {{"skills": 0.4, "experience": 0.35, "education": 0.15, "projects": 0.1}},
  "hard_requirements": ["<must-have 1>", "<must-have 2>"],
  "nice_to_have": ["<optional 1>"],
  "top_candidate_count": 5
}}"""

        try:
            resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.2)
            plan = _parse_json_response(resp)
            return {
                "plan": plan,
                "search_queries": plan.get("search_queries", [state['job_description']])
            }
        except Exception as e:
            return {"plan": {}, "search_queries": [state['job_description']], "errors": [f"Planner error: {e}"]}

    # ── Retriever ─────────────────────────────────────────────────────────────
    def retriever_node(self, state: PipelineState) -> dict:
        """Retrieve ALL candidates from ChromaDB, ranked by semantic similarity."""
        pool_size = self.store.count()
        if pool_size == 0:
            return {"retrieved_ids": [], "errors": ["No candidates indexed in ChromaDB"]}

        # Fetch every candidate for each query (top_k = full pool), then dedup.
        # This guarantees newly uploaded resumes are always evaluated regardless
        # of how they rank against existing candidates.
        seen = set()
        results = []
        for query in state.get("search_queries", []):
            for hit in self.store.search(query, top_k=pool_size):
                if hit["id"] not in seen:
                    seen.add(hit["id"])
                    results.append(hit)

        # Sort by best similarity score seen across all queries
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return {"retrieved_ids": [r["id"] for r in results]}

    # ── Evaluator ─────────────────────────────────────────────────────────────
    def evaluator_node(self, state: PipelineState) -> dict:
        """Score each retrieved candidate against the JD — parallel LLM calls."""
        plan = state.get("plan", {})
        retrieved_ids = state.get("retrieved_ids", [])

        def eval_one(cid: str) -> dict:
            if cid in self.candidates:
                resume_dict = self.candidates[cid].dict()
                ci = self.candidates[cid].contact_info
                name = (ci.name if ci else None) or cid
            else:
                resume_dict = self.store.get_candidate_data(cid)
                if not resume_dict:
                    return None
                ci = resume_dict.get("contact_info") or {}
                name = ci.get("name") or cid

            prompt = f"""You are an expert recruiter. Evaluate this candidate against the job description.

Job Description:
{state['job_description']}

Scoring weights: {json.dumps(plan.get('scoring_weights', {}))}
Hard requirements: {json.dumps(plan.get('hard_requirements', []))}
Nice to have: {json.dumps(plan.get('nice_to_have', []))}

Candidate Resume (JSON):
{json.dumps(resume_dict, indent=2)}

Return ONLY valid JSON:
{{
  "match_score": <float 0-100>,
  "skills_score": <float 0-100>,
  "experience_score": <float 0-100>,
  "education_score": <float 0-100>,
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<gap1>", "<gap2>"],
  "matched_skills": ["<skill1>"],
  "missing_skills": ["<skill1>"],
  "meets_hard_requirements": <true|false>,
  "overall_recommendation": "<STRONG_MATCH|MATCH|WEAK_MATCH|NOT_QUALIFIED>",
  "reasoning": "<1-2 sentence summary>",
  "suggested_interview_questions": ["<q1>", "<q2>"]
}}"""

            try:
                resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.2)
                ev = _parse_json_response(resp)
                ev["candidate_id"] = cid
                ev["candidate_name"] = name
                return ev
            except Exception as e:
                return {"candidate_id": cid, "candidate_name": name,
                        "match_score": 0, "overall_recommendation": "ERROR", "reasoning": str(e)}

        evaluations = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(eval_one, cid): cid for cid in retrieved_ids}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    evaluations.append(result)

        return {"evaluations": evaluations}

    # ── Critique ──────────────────────────────────────────────────────────────
    def critique_node(self, state: PipelineState) -> dict:
        """Validate evaluations for consistency and flag any issues"""
        evaluations = state.get("evaluations", [])
        if not evaluations:
            return {"critique": {"is_valid": False, "issues": ["No candidates evaluated"]}}

        prompt = f"""You are a QA reviewer for a recruitment pipeline. Review these candidate evaluations for consistency.

Job Description (summary):
{state['job_description'][:500]}

Evaluations:
{json.dumps(evaluations, indent=2)}

Check for: score inflation/deflation, ignored hard requirements, inconsistent recommendations.

Return ONLY valid JSON:
{{
  "is_valid": <true|false>,
  "issues_found": ["<issue1>"],
  "adjustments": [{{"candidate_id": "<id>", "adjusted_score": <float>, "reason": "<why>"}}],
  "recommended_for_interview": ["<candidate_id1>", "<candidate_id2>"],
  "overall_quality": "<GOOD|ACCEPTABLE|NEEDS_REVIEW>",
  "summary": "<1-2 sentence summary of the candidate pool>"
}}"""

        try:
            resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.2)
            critique = _parse_json_response(resp)

            # Apply score adjustments from critique
            adjustments = {a["candidate_id"]: a["adjusted_score"] for a in critique.get("adjustments", [])}
            for ev in evaluations:
                cid = ev.get("candidate_id")
                if cid in adjustments:
                    ev["match_score"] = adjustments[cid]
                    ev["score_adjusted"] = True

            return {"critique": critique}
        except Exception as e:
            return {"critique": {"is_valid": False, "issues_found": [f"Critique error: {e}"]}, "errors": [str(e)]}

    # ── Synthesizer ───────────────────────────────────────────────────────────
    def synthesizer_node(self, state: PipelineState) -> dict:
        """Produce final ranked list"""
        evaluations = state.get("evaluations", [])
        critique = state.get("critique", {})
        recommended_ids = set(critique.get("recommended_for_interview", []))

        ranked = sorted(evaluations, key=lambda x: x.get("match_score", 0), reverse=True)
        for i, ev in enumerate(ranked):
            ev["rank"] = i + 1
            ev["recommended_for_interview"] = ev.get("candidate_id") in recommended_ids

        return {"final_rankings": ranked}

    # ── Interview Question Generator ──────────────────────────────────────────
    def interview_questions_node(self, state: PipelineState) -> dict:
        """Generate tailored interview questions for recommended candidates — parallel."""
        recommended = [r for r in state.get("final_rankings", []) if r.get("recommended_for_interview")]

        def gen_questions(ranking: dict) -> tuple:
            cid = ranking["candidate_id"]
            if cid in self.candidates:
                resume_dict = self.candidates[cid].dict()
            else:
                resume_dict = self.store.get_candidate_data(cid)
                if not resume_dict:
                    return cid, {"error": "Resume data not found"}

            prompt = f"""You are a senior technical interviewer. Generate a tailored interview question set for this candidate.

Role being hired for:
{state['job_description']}

Candidate resume:
{json.dumps(resume_dict, indent=2)}

Screening evaluation:
- Match score: {ranking.get('match_score')}/100
- Strengths: {json.dumps(ranking.get('strengths', []))}
- Weaknesses / gaps: {json.dumps(ranking.get('weaknesses', []))}
- Missing skills: {json.dumps(ranking.get('missing_skills', []))}
- Matched skills: {json.dumps(ranking.get('matched_skills', []))}

Generate questions that:
1. Probe depth on claimed strengths (not surface-level)
2. Directly challenge each identified gap or missing skill
3. Dig into specific projects they listed (not generic)
4. Include at least one system design / architecture question relevant to the role
5. Include one behavioral question tied to a real gap (e.g. "Tell me about leading a team" if leadership is missing)

Return ONLY valid JSON:
{{
  "candidate_name": "<name>",
  "technical_depth": [
    {{"question": "<q>", "what_to_look_for": "<expected signal>", "follow_up": "<follow up if answer is shallow>"}}
  ],
  "gap_probing": [
    {{"gap": "<missing skill or weakness>", "question": "<q>", "what_to_look_for": "<signal>"}}
  ],
  "project_specific": [
    {{"project": "<project name from resume>", "question": "<q>", "what_to_look_for": "<signal>"}}
  ],
  "system_design": [
    {{"question": "<q>", "what_to_look_for": "<signal>"}}
  ],
  "behavioral": [
    {{"question": "<q>", "what_to_look_for": "<signal>"}}
  ]
}}"""

            try:
                resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=1400)
                questions = _parse_json_response(resp)
                questions["candidate_id"] = cid
                questions["rank"] = ranking["rank"]
                questions["match_score"] = ranking.get("match_score")
                return cid, questions
            except Exception as e:
                return cid, {"error": str(e)}

        # Cap at top 5 recommended to keep IQ generation fast
        recommended = recommended[:5]

        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(gen_questions, r) for r in recommended]
            for future in as_completed(futures):
                cid, questions = future.result()
                results[cid] = questions

        return {"interview_questions": results}


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_screening_graph(llm: OpenRouterClient, store: CandidateVectorStore, candidate_store: Dict[str, Resume]):
    """Construct and compile the LangGraph screening pipeline"""
    nodes = AgentNodes(llm, store, candidate_store)

    graph = StateGraph(PipelineState)
    graph.add_node("planner", nodes.planner_node)
    graph.add_node("retriever", nodes.retriever_node)
    graph.add_node("evaluator", nodes.evaluator_node)
    graph.add_node("critique", nodes.critique_node)
    graph.add_node("synthesizer", nodes.synthesizer_node)
    graph.add_node("interview_questions", nodes.interview_questions_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "evaluator")
    graph.add_edge("evaluator", "critique")
    graph.add_edge("critique", "synthesizer")
    graph.add_edge("synthesizer", "interview_questions")
    graph.add_edge("interview_questions", END)

    return graph.compile()


# ─── Pipeline Facade ──────────────────────────────────────────────────────────

class ScreeningPipeline:
    def __init__(self, openrouter_api_key: Optional[str] = None, chroma_path: str = "./chroma_db"):
        self.llm = OpenRouterClient(openrouter_api_key)
        self.store = CandidateVectorStore(chroma_path)
        self.candidate_store: Dict[str, Resume] = {}
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            self._graph = build_screening_graph(self.llm, self.store, self.candidate_store)
        return self._graph

    def add_candidate(self, resume: Resume, candidate_id: str):
        self.candidate_store[candidate_id] = resume
        self.store.add_candidate(resume, candidate_id)
        self._graph = None  # Rebuild graph with updated store

    def screen_candidates(self, query: ScreeningQuery) -> dict:
        """Run the full LangGraph pipeline and return state"""
        graph = self._get_graph()

        initial_state: PipelineState = {
            "job_description": query.job_description,
            "required_skills": query.required_skills or [],
            "min_experience_years": query.min_experience_years,
            "required_education": query.required_education,
            "evaluation_criteria": query.evaluation_criteria,
            "plan": None,
            "search_queries": [],
            "retrieved_ids": [],
            "evaluations": [],
            "critique": None,
            "final_rankings": [],
            "interview_questions": {},
            "errors": []
        }

        return graph.invoke(initial_state)
