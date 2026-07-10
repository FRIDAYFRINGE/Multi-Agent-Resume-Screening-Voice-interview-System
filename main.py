from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import Optional, List
import json

from resume_parser import ResumeParser
from openrouter_client import OpenRouterClient
from agentic_pipeline import ScreeningPipeline, ScreeningQuery
from schemas import Resume

app = FastAPI(
    title="Resume Screening Assistant",
    description="Multi-agent RAG system for resume screening",
    version="1.0.0"
)

# Initialize components
parser = ResumeParser()
llm_client = None
pipeline = None


@app.on_event("startup")
async def startup():
    """Initialize LLM and pipeline on startup"""
    global llm_client, pipeline

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set. Some features will be limited.")
    else:
        llm_client = OpenRouterClient(api_key)
        pipeline = ScreeningPipeline(api_key)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openrouter_configured": llm_client is not None
    }


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    candidate_id: Optional[str] = None
):
    """Upload and parse a resume"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not configured")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Parse resume
        resume = parser.parse_resume(tmp_path, llm_client)

        # Add to pipeline
        if pipeline:
            cid = candidate_id or (resume.contact_info.name or f"candidate_{abs(hash(str(resume)))}")
            pipeline.add_candidate(resume, cid)

        return {
            "status": "success",
            "candidate_id": cid,
            "resume": resume.dict(),
            "message": "Resume parsed and indexed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing resume: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/parse-resume-text")
async def parse_resume_text(
    text: str,
    candidate_id: Optional[str] = None
):
    """Parse resume from raw text"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not configured")

    try:
        resume = llm_client.extract_resume_from_text(text)

        # Add to pipeline
        if pipeline:
            cid = candidate_id or (resume.contact_info.name or "unnamed_candidate")
            pipeline.add_candidate(resume, cid)

        return {
            "status": "success",
            "candidate_id": cid,
            "resume": resume.dict(),
            "message": "Resume parsed from text successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing resume: {str(e)}")


@app.post("/screen-candidates")
async def screen_candidates(query: ScreeningQuery):
    """Run screening pipeline on all uploaded candidates"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        state = pipeline.screen_candidates(query)

        return {
            "status": "success",
            "total_candidates": len(state.candidates),
            "retrieved_count": len(state.retrieved_candidates),
            "evaluated_count": len(state.evaluations),
            "plan": state.plan,
            "final_rankings": state.final_rankings[:10],  # Top 10
            "critique_feedback": state.critique_feedback,
            "errors": state.errors
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error screening candidates: {str(e)}")


@app.get("/candidates")
async def list_candidates():
    """List all uploaded candidates"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    candidates = []
    for cid, resume in pipeline.candidate_store.items():
        candidates.append({
            "id": cid,
            "name": resume.contact_info.name if resume.contact_info else "Unknown",
            "email": resume.contact_info.email if resume.contact_info else None,
            "skills": [s.skills for s in resume.skills]
        })

    return {"candidates": candidates, "total": len(candidates)}


@app.get("/candidate/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Get detailed resume of a specific candidate"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if candidate_id not in pipeline.candidate_store:
        raise HTTPException(status_code=404, detail="Candidate not found")

    resume = pipeline.candidate_store[candidate_id]
    return {
        "candidate_id": candidate_id,
        "resume": resume.dict()
    }


@app.post("/evaluate-candidate/{candidate_id}")
async def evaluate_candidate(
    candidate_id: str,
    job_description: str,
    evaluation_criteria: Optional[dict] = None
):
    """Evaluate a specific candidate against a job description"""
    if not llm_client or not pipeline:
        raise HTTPException(status_code=503, detail="Services not configured")

    if candidate_id not in pipeline.candidate_store:
        raise HTTPException(status_code=404, detail="Candidate not found")

    try:
        resume = pipeline.candidate_store[candidate_id]
        evaluation = llm_client.evaluate_candidate(
            resume,
            job_description,
            evaluation_criteria
        )

        return {
            "candidate_id": candidate_id,
            "evaluation": evaluation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating candidate: {str(e)}")


@app.post("/bulk-upload")
async def bulk_upload(files: List[UploadFile] = File(...)):
    """Upload multiple resumes at once"""
    if not llm_client or not pipeline:
        raise HTTPException(status_code=503, detail="Services not configured")

    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            resume = parser.parse_resume(tmp_path, llm_client)
            cid = resume.contact_info.name or f"candidate_{abs(hash(str(resume)))}"
            pipeline.add_candidate(resume, cid)

            results.append({
                "filename": file.filename,
                "status": "success",
                "candidate_id": cid
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
        finally:
            os.unlink(tmp_path)

    return {
        "total_uploaded": len(files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
