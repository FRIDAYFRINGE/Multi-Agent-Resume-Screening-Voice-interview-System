import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from openrouter_client import OpenRouterClient
from schemas import Resume
import chromadb
from chromadb.config import Settings


class ScreeningQuery(BaseModel):
    job_description: str
    required_skills: List[str] = []
    min_experience_years: Optional[int] = None
    required_education: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None


class ScreeningPipelineState(BaseModel):
    query: ScreeningQuery
    candidates: List[Resume] = []
    plan: Optional[Dict[str, Any]] = None
    retrieved_candidates: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    final_rankings: List[Dict[str, Any]] = []
    critique_feedback: Optional[str] = None
    errors: List[str] = []


class PlannerAgent:
    """Decomposes screening queries into search and evaluation sub-tasks"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm = llm_client

    def plan(self, query: ScreeningQuery) -> Dict[str, Any]:
        """Create a plan for candidate screening"""
        prompt = f"""
You are a screening coordinator. Create a detailed plan for screening candidates.

Job Description:
{query.job_description}

Required Skills:
{', '.join(query.required_skills) if query.required_skills else 'Not specified'}

Minimum Experience: {query.min_experience_years} years
Required Education: {query.required_education or 'Not specified'}

Create a JSON plan with:
{{
    "search_queries": [string],  // Semantic search queries
    "evaluation_focus_areas": [string],  // Key areas to evaluate
    "scoring_weights": {{"skills": float, "experience": float, "education": float}},
    "rejection_criteria": [string],  // Auto-reject if any match
    "top_candidate_count": int
}}

Return ONLY valid JSON.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.call_llm(messages, temperature=0.3)

        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()

        return json.loads(json_str)


class RetrievalAgent:
    """Searches and retrieves candidate embeddings from ChromaDB"""

    def __init__(self, collection_name: str = "candidates"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb",
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_candidate(self, resume: Resume, candidate_id: str, metadata: Optional[Dict] = None):
        """Add candidate resume to vector store"""
        # Create text representation for embedding
        text = self._resume_to_text(resume)

        embed_metadata = metadata or {}
        embed_metadata["candidate_id"] = candidate_id
        embed_metadata["name"] = resume.contact_info.name if resume.contact_info else "Unknown"

        self.collection.add(
            ids=[candidate_id],
            documents=[text],
            metadatas=[embed_metadata]
        )

    def search_candidates(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for candidates using semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )

        candidates = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i, candidate_id in enumerate(results['ids'][0]):
                candidates.append({
                    "id": candidate_id,
                    "similarity": results['distances'][0][i] if results['distances'] else None,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return candidates

    def _resume_to_text(self, resume: Resume) -> str:
        """Convert resume to searchable text"""
        parts = []

        if resume.professional_summary:
            parts.append(resume.professional_summary)

        for skill_cat in resume.skills:
            if skill_cat.skills:
                parts.append(" ".join(skill_cat.skills))

        for exp in resume.experience:
            if exp.description:
                parts.append(exp.description)
            if exp.technologies:
                parts.append(" ".join(exp.technologies))

        for edu in resume.education:
            if edu.field_of_study:
                parts.append(edu.field_of_study)
            if edu.description:
                parts.append(edu.description)

        return " ".join(parts)


class CritiqueAgent:
    """Validates evaluations against JD requirements"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm = llm_client

    def validate_evaluations(
        self,
        evaluations: List[Dict[str, Any]],
        job_description: str,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Critique and validate candidate evaluations"""

        prompt = f"""
You are a quality assurance expert reviewing candidate evaluations.

Job Description:
{job_description}

Screening Plan:
{json.dumps(plan, indent=2)}

Candidate Evaluations:
{json.dumps(evaluations[:5], indent=2)}  // First 5 for brevity

Validate these evaluations. Return a JSON with:
{{
    "is_valid": bool,
    "issues": [string],  // Issues found
    "recommendations": [string],  // Suggested adjustments
    "top_candidates": [string],  // IDs of top candidates to interview
    "pass_rate": float,  // Percentage of viable candidates
    "feedback": string  // Overall feedback
}}

Return ONLY valid JSON.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.call_llm(messages, temperature=0.3)

        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()

        return json.loads(json_str)


class ScreeningPipeline:
    """Main agentic screening pipeline orchestrator"""

    def __init__(self, openrouter_api_key: Optional[str] = None):
        self.llm = OpenRouterClient(openrouter_api_key)
        self.planner = PlannerAgent(self.llm)
        self.retriever = RetrievalAgent()
        self.critic = CritiqueAgent(self.llm)
        self.candidate_store: Dict[str, Resume] = {}

    def add_candidate(self, resume: Resume, candidate_id: str):
        """Add candidate to pipeline"""
        self.candidate_store[candidate_id] = resume
        self.retriever.add_candidate(resume, candidate_id)

    def screen_candidates(self, query: ScreeningQuery) -> ScreeningPipelineState:
        """Run full screening pipeline"""
        state = ScreeningPipelineState(query=query)
        state.candidates = list(self.candidate_store.values())

        try:
            # Step 1: Planning
            state.plan = self.planner.plan(query)

            # Step 2: Retrieval - semantic search using ChromaDB
            search_queries = state.plan.get("search_queries", [])
            all_retrieved = set()
            for search_query in search_queries:
                results = self.retriever.search_candidates(search_query, top_k=20)
                all_retrieved.update([r["id"] for r in results])
            state.retrieved_candidates = list(all_retrieved)

            # Step 3: Evaluation - score candidates
            for candidate_id in state.retrieved_candidates:
                if candidate_id in self.candidate_store:
                    resume = self.candidate_store[candidate_id]
                    evaluation = self.llm.evaluate_candidate(
                        resume,
                        query.job_description,
                        query.evaluation_criteria
                    )
                    evaluation["candidate_id"] = candidate_id
                    state.evaluations.append(evaluation)

            # Step 4: Critique - validate results
            if state.evaluations:
                critique = self.critic.validate_evaluations(
                    state.evaluations,
                    query.job_description,
                    state.plan
                )
                state.critique_feedback = critique.get("feedback", "")

                # Step 5: Ranking - final rankings
                state.final_rankings = self._rank_candidates(state.evaluations, critique)

        except Exception as e:
            state.errors.append(f"Pipeline error: {str(e)}")

        return state

    def _rank_candidates(
        self,
        evaluations: List[Dict[str, Any]],
        critique: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank candidates based on evaluations"""
        # Sort by match score
        ranked = sorted(
            evaluations,
            key=lambda x: x.get("match_score", 0),
            reverse=True
        )

        # Add rank
        for i, candidate in enumerate(ranked):
            candidate["rank"] = i + 1

        return ranked
