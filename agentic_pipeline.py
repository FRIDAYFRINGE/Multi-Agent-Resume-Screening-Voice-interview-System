import json
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
        """Use ChromaDB semantic search to retrieve relevant candidates"""
        if self.store.count() == 0:
            return {"retrieved_ids": [], "errors": ["No candidates indexed in ChromaDB"]}

        seen = set()
        results = []
        for query in state.get("search_queries", []):
            for hit in self.store.search(query, top_k=10):
                if hit["id"] not in seen:
                    seen.add(hit["id"])
                    results.append(hit)

        # Sort by similarity descending, take top 20
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        retrieved_ids = [r["id"] for r in results[:20]]

        return {"retrieved_ids": retrieved_ids}

    # ── Evaluator ─────────────────────────────────────────────────────────────
    def evaluator_node(self, state: PipelineState) -> dict:
        """Score each retrieved candidate against the JD"""
        evaluations = []
        plan = state.get("plan", {})

        for cid in state.get("retrieved_ids", []):
            if cid not in self.candidates:
                continue
            resume = self.candidates[cid]

            prompt = f"""You are an expert recruiter. Evaluate this candidate against the job description.

Job Description:
{state['job_description']}

Scoring weights: {json.dumps(plan.get('scoring_weights', {}))}
Hard requirements: {json.dumps(plan.get('hard_requirements', []))}
Nice to have: {json.dumps(plan.get('nice_to_have', []))}

Candidate Resume (JSON):
{json.dumps(resume.dict(), indent=2)}

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
                evaluation = _parse_json_response(resp)
                evaluation["candidate_id"] = cid
                evaluation["candidate_name"] = resume.contact_info.name if resume.contact_info else cid
                evaluations.append(evaluation)
            except Exception as e:
                evaluations.append({
                    "candidate_id": cid,
                    "match_score": 0,
                    "overall_recommendation": "ERROR",
                    "reasoning": str(e)
                })

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
        """Generate tailored interview questions for each recommended candidate"""
        results: Dict[str, Any] = {}

        for ranking in state.get("final_rankings", []):
            if not ranking.get("recommended_for_interview"):
                continue

            cid = ranking["candidate_id"]
            resume = self.candidates.get(cid)
            if not resume:
                continue

            prompt = f"""You are a senior technical interviewer. Generate a tailored interview question set for this candidate.

Role being hired for:
{state['job_description']}

Candidate resume:
{json.dumps(resume.dict(), indent=2)}

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
                resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=2048)
                questions = _parse_json_response(resp)
                questions["candidate_id"] = cid
                questions["rank"] = ranking["rank"]
                questions["match_score"] = ranking.get("match_score")
                results[cid] = questions
            except Exception as e:
                results[cid] = {"error": str(e)}

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
