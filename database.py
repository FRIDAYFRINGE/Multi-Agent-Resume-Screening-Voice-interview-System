"""
MongoDB integration — persistent storage for candidates, job screening runs,
and interview reports.

Collections:
  candidates        — one doc per person, keyed by email
  jobs              — one doc per screening run (JD + full rankings)
  interview_reports — one doc per completed interview (transcript + verdict)

Candidate identity
------------------
Email is the only identity. One address → one candidate → one resume.

  _id       : the lowercased email address (MongoDB primary key)
  chroma_id : slug of the email local-part  (john.doe@gmail.com → john_doe)

Re-uploading always overwrites in place; there are no resume versions and no
name-derived fallback. A resume with no readable email cannot be identified and
is rejected by the caller rather than stored under a guessed id.
"""
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient, DESCENDING

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB",  "interview_agentic")

_client: Optional[MongoClient] = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _client[DB_NAME]


def is_connected() -> bool:
    try:
        get_db().command("ping")
        return True
    except Exception:
        return False


# ── ID helpers ────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert arbitrary text to a lowercase underscore slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "candidate"


def _email_to_base_id(email: str) -> str:
    local = email.split("@")[0]
    return _slugify(local)


def chroma_id_for_email(email: str) -> str:
    """The one ChromaDB id belonging to this address.

    The slug comes from the local-part only, so two different addresses sharing
    one local-part collide — including the case that actually bit us, a PDF
    extraction dropping a character from the domain
    ("...@gmail.com" vs "...@gmail.co"). When the plain slug already belongs to a
    *different* address, a short digest of the full address is appended so the two
    stay distinct. Addresses that already own their slug keep it unchanged, so
    stored ids are stable and nothing needs re-embedding.
    """
    email = (email or "").strip().lower()
    base = _email_to_base_id(email)
    try:
        owner = get_db().candidates.find_one(
            {"chroma_id": base}, {"_id": 1}
        )
    except Exception:
        return base  # Mongo unavailable — cannot check, keep the simple id
    if not owner or str(owner["_id"]) == email:
        return base
    digest = hashlib.sha256(email.encode()).hexdigest()[:6]
    return f"{base}_{digest}"


# ── Candidates ────────────────────────────────────────────────────────────────

def find_candidate_by_email(email: str) -> Optional[dict]:
    """
    Return the stored candidate document for this email, or None.

    One email → one resume.  The candidate doc's _id *is* the lowercased email,
    so this is an exact primary-key hit rather than a similarity guess — it
    costs one indexed lookup and no LLM call, which is what lets the caller
    decide whether to re-parse before spending anything.

    The returned dict carries a "chroma_id" key pointing at the stored resume.
    """
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return None

    _PROJECTION = {"name": 1, "email": 1, "chroma_id": 1}
    db = get_db()

    try:
        hit = db.candidates.find_one({"_id": email}, _PROJECTION)
        if hit:
            return hit

        # PDF text extraction can fuse neighbouring glyph text onto the front of
        # the address — LaTeX icon fonts are the usual culprit, e.g. the
        # "envelope" icon leaving "…envel⌢pe" glued to the real local-part.
        # That corruption only ever *prepends*, so any stored address the
        # extracted string ends on belongs to the same person.
        domain = email.rpartition("@")[2]
        if not domain:
            return None
        for cand in db.candidates.find(
            {"email": {"$regex": re.escape("@" + domain) + "$"}}, _PROJECTION
        ).limit(500):
            stored = (cand.get("email") or "").strip().lower()
            if stored and email.endswith(stored):
                return cand
    except Exception:
        return None
    return None


def upsert_candidate(
    email: str,
    name: str,
    filename: str,
    resume_dict: dict,
    source: str = "uploaded",
) -> str:
    """Store the one resume belonging to this address, replacing any previous one.

    Returns the chroma_id the resume is indexed under. Re-uploading the same
    address overwrites in place, so a person can never accumulate documents.
    """
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("upsert_candidate requires an email address")

    chroma_id = chroma_id_for_email(email)
    now = datetime.now(timezone.utc)
    get_db().candidates.update_one(
        {"_id": email},
        {
            "$set": {
                "chroma_id":   chroma_id,
                "name":        name,
                "email":       email,
                "source":      source,
                "filename":    filename or "",
                "resume_json": resume_dict,
                "updated_at":  now,
            },
            "$setOnInsert": {"added_at": now},
        },
        upsert=True,
    )
    return chroma_id


def save_candidate(candidate_id: str, resume_dict: dict, source: str = "seeded") -> None:
    """Upsert used by seed_candidates.py. Seeds are email-keyed like everything else."""
    ci    = resume_dict.get("contact_info") or {}
    email = (ci.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError(f"seed candidate {candidate_id!r} has no email address")
    upsert_candidate(
        email, ci.get("name") or candidate_id, "seeded", resume_dict, source=source
    )


def get_candidate(candidate_id: str) -> Optional[dict]:
    db = get_db()
    return db.candidates.find_one({"_id": candidate_id})


def list_candidates(limit: int = 200) -> list:
    db = get_db()
    return list(db.candidates.find({}, {"resume_json": 0}).limit(limit))


def candidate_count() -> int:
    return get_db().candidates.count_documents({})


# ── Screening Jobs ────────────────────────────────────────────────────────────

def make_jd_hash(jd_text: str) -> str:
    """16-char hex fingerprint of a normalised JD — used to detect re-runs."""
    normalised = re.sub(r"\s+", " ", (jd_text or "").lower().strip())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def find_job_by_hash(jd_hash: str) -> Optional[dict]:
    """Return the most recent job run whose JD matches this hash, or None."""
    try:
        return get_db().jobs.find_one(
            {"jd_hash": jd_hash},
            sort=[("run_at", DESCENDING)]
        )
    except Exception:
        return None


def save_job(job_doc: dict) -> str:
    """Insert a screening run and return its string id."""
    db = get_db()
    result = db.jobs.insert_one({
        "jd_hash":              job_doc.get("jd_hash", ""),
        "job_description":      job_doc["job_description"],
        "required_skills":      job_doc.get("required_skills", []),
        "pool_size":            job_doc.get("pool_size", 0),
        "pool_fingerprint":     job_doc.get("pool_fingerprint", ""),
        "pool_manifest":        job_doc.get("pool_manifest", {}),
        "plan":                 job_doc.get("plan", {}),
        "screening_stats":      job_doc.get("screening_stats", {}),
        "candidates_evaluated": (job_doc.get("screening_stats") or {}).get(
            "llm_evaluated_count", len(job_doc.get("rankings", []))
        ),
        "rankings":             job_doc.get("rankings", []),
        "critique_summary":     (job_doc.get("critique") or {}).get("summary", ""),
        "run_at":               datetime.now(timezone.utc),
    })
    return str(result.inserted_id)


def update_job_rankings(job_id, rankings: list, pool_size: int) -> None:
    """
    Write merged rankings back onto an existing job.

    The fast path evaluates candidates the cached run never saw; without this
    that work is discarded and the next re-run pays for it again.
    """
    from bson import ObjectId  # local import — pymongo is optional at import time
    try:
        oid = ObjectId(str(job_id))
    except Exception:
        return
    get_db().jobs.update_one(
        {"_id": oid},
        {"$set": {
            "rankings":   rankings,
            "pool_size":  pool_size,
            "updated_at": datetime.now(timezone.utc),
        }},
    )


def get_plan_for_hash(jd_hash: str) -> Optional[dict]:
    """Return cached planner output for a JD hash, or None."""
    try:
        doc = get_db().jd_plans.find_one({"_id": jd_hash})
        return doc.get("plan") if doc else None
    except Exception:
        return None


def save_plan_for_hash(jd_hash: str, plan: dict) -> None:
    """Cache planner output keyed by JD hash."""
    try:
        get_db().jd_plans.update_one(
            {"_id": jd_hash},
            {"$set": {"plan": plan, "saved_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    except Exception:
        pass


def list_jobs(limit: int = 50) -> list:
    db = get_db()
    return list(
        db.jobs.find(
            {},
            {"job_description": 1, "required_skills": 1, "pool_size": 1,
             "candidates_evaluated": 1, "critique_summary": 1, "run_at": 1}
        ).sort("run_at", DESCENDING).limit(limit)
    )


def get_job(job_id: str) -> Optional[dict]:
    from bson import ObjectId
    db = get_db()
    return db.jobs.find_one({"_id": ObjectId(job_id)})


# ── Interview Reports ─────────────────────────────────────────────────────────

def save_interview_report(
    session_id: str,
    job_id: Optional[str],
    candidate_id: str,
    candidate_name: str,
    job_description: str,
    transcript: list,
    report: dict,
) -> None:
    db = get_db()
    db.interview_reports.update_one(
        {"_id": session_id},
        {"$set": {
            "_id":            session_id,
            "job_id":         job_id,
            "candidate_id":   candidate_id,
            "candidate_name": candidate_name,
            "job_description": job_description[:500],
            "verdict":        report.get("verdict"),
            "overall_score":  report.get("overall_score"),
            "summary":        report.get("summary"),
            "transcript":     transcript,
            "report":         report,
            "completed_at":   datetime.now(timezone.utc),
        }},
        upsert=True,
    )


def list_reports(limit: int = 50) -> list:
    db = get_db()
    return list(
        db.interview_reports.find(
            {},
            {"candidate_name": 1, "verdict": 1, "overall_score": 1,
             "job_description": 1, "completed_at": 1}
        ).sort("completed_at", DESCENDING).limit(limit)
    )


def get_report_by_session(session_id: str) -> Optional[dict]:
    """Return the full interview report document for a given session ID."""
    try:
        return get_db().interview_reports.find_one({"_id": session_id})
    except Exception:
        return None


# ── Session Persistence ───────────────────────────────────────────────────────

def _strip_session(session: dict) -> dict:
    """Remove non-serialisable runtime keys before saving to MongoDB."""
    return {k: v for k, v in session.items() if k not in ("ws", "stt", "local_stt")}


def save_session(session_id: str, session: dict) -> None:
    """Upsert a full session document (initial save on creation)."""
    try:
        doc = _strip_session(session)
        doc.setdefault("status", "active")
        get_db().sessions.update_one(
            {"_id": session_id},
            {"$set": {**doc, "_id": session_id, "updated_at": datetime.now(timezone.utc)},
             "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    except Exception:
        pass


def update_session_transcript(
    session_id: str,
    current_idx: int,
    follow_up_count: int,
    transcript: list,
    answer_scores: list,
) -> None:
    """Partial update after each answered question — keeps transcript in sync."""
    try:
        get_db().sessions.update_one(
            {"_id": session_id},
            {"$set": {
                "current_idx":     current_idx,
                "follow_up_count": follow_up_count,
                "transcript":      transcript,
                "answer_scores":   answer_scores,
                "updated_at":      datetime.now(timezone.utc),
            }},
        )
    except Exception:
        pass


def mark_session_completed(session_id: str) -> None:
    """Mark a session as completed once the final report is generated."""
    try:
        get_db().sessions.update_one(
            {"_id": session_id},
            {"$set": {"status": "completed", "updated_at": datetime.now(timezone.utc)}},
        )
    except Exception:
        pass


def load_session(session_id: str) -> Optional[dict]:
    """Restore a session from MongoDB (used after server restart)."""
    try:
        doc = get_db().sessions.find_one({"_id": session_id})
        if doc:
            doc.pop("_id", None)
            doc.pop("created_at", None)
            doc.pop("updated_at", None)
            doc.pop("status", None)
            return doc
    except Exception:
        pass
    return None
