"""
MongoDB integration — persistent storage for candidates, job screening runs,
and interview reports.

Collections:
  candidates        — one doc per person (keyed by email or name-slug)
                      resumes[] array holds every version uploaded
  jobs              — one doc per screening run (JD + full rankings)
  interview_reports — one doc per completed interview (transcript + verdict)

Candidate ID scheme
-------------------
  base_id   : slug derived from email local-part  (e.g. john.doe@gmail.com → john_doe)
              falls back to slugified name if no email
  chroma_id : base_id + "_" + sub_id              (e.g. john_doe_1, john_doe_2)
  doc_key   : email if available, else base_id    (used as MongoDB _id)

Each uploaded resume for the same person gets an incrementing sub_id so all
versions are stored and indexed independently in ChromaDB.
"""
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Optional, Tuple
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


def resolve_candidate_id(
    email: str,
    name: str,
    chroma_store=None,
) -> Tuple[str, str, int, str]:
    """
    Determine the correct (base_id, chroma_id, sub_id, doc_key) for a new
    resume, handling duplicates by incrementing sub_id.

    Priority for sub_id lookup:
      1. MongoDB (authoritative, if connected)
      2. ChromaDB scan (fallback when MongoDB is down)
      3. Default to 1 (first ever resume for this person)

    Returns
    -------
    base_id   : clean slug (e.g. "john_doe")
    chroma_id : base_id + "_" + sub_id  (e.g. "john_doe_2")
    sub_id    : integer starting at 1
    doc_key   : MongoDB _id to use (email preferred, else base_id)
    """
    email = (email or "").strip().lower()

    if email and "@" in email:
        base_id = _email_to_base_id(email)
        doc_key = email
    else:
        base_id = _slugify(name or "candidate")
        doc_key = base_id
        email = ""

    sub_id = 1

    # 1. MongoDB lookup
    if is_connected():
        try:
            existing = get_db().candidates.find_one(
                {"_id": doc_key}, {"latest_sub_id": 1}
            )
            if existing:
                sub_id = existing.get("latest_sub_id", 0) + 1
            return base_id, f"{base_id}_{sub_id}", sub_id, doc_key
        except Exception:
            pass

    # 2. ChromaDB fallback — scan for base_id_1, base_id_2, …
    if chroma_store is not None:
        try:
            n = 0
            while True:
                probe = f"{base_id}_{n + 1}"
                result = chroma_store.collection.get(ids=[probe], include=[])
                if result and result.get("ids"):
                    n += 1
                else:
                    break
            sub_id = n + 1
        except Exception:
            pass

    return base_id, f"{base_id}_{sub_id}", sub_id, doc_key


# ── Candidates ────────────────────────────────────────────────────────────────

def find_resume_by_filename(doc_key: str, filename: str) -> Optional[dict]:
    """
    Return the resumes[] entry whose filename matches, or None.
    Used to detect duplicate / updated uploads before assigning a new sub_id.
    """
    if not filename:
        return None
    try:
        doc = get_db().candidates.find_one(
            {"_id": doc_key, "resumes.filename": filename},
            {"resumes.$": 1}
        )
        if doc and doc.get("resumes"):
            return doc["resumes"][0]
    except Exception:
        pass
    return None


def update_resume_version_inplace(
    doc_key: str,
    chroma_id: str,
    resume_dict: dict,
) -> None:
    """
    Overwrite the data for an existing resume version (same filename, changed
    content — similarity < 90%).  Updates resume_json on the top-level doc and
    the timestamp on the matching resumes[] entry; does NOT push a new entry.
    """
    db = get_db()
    now = datetime.now(timezone.utc)
    db.candidates.update_one(
        {"_id": doc_key, "resumes.chroma_id": chroma_id},
        {
            "$set": {
                "resume_json":            resume_dict,
                "updated_at":             now,
                "resumes.$.added_at":     now,
            }
        }
    )


def add_resume_version(
    doc_key: str,
    base_id: str,
    email: str,
    name: str,
    chroma_id: str,
    sub_id: int,
    filename: str,
    resume_dict: dict,
    source: str = "uploaded",
) -> None:
    """
    Upsert the candidate document (keyed by email or base_id) and push a new
    resume version entry.  Multiple resumes for the same person accumulate in
    the resumes[] array; latest_sub_id tracks the highest version.
    """
    db = get_db()
    now = datetime.now(timezone.utc)

    db.candidates.update_one(
        {"_id": doc_key},
        {
            "$set": {
                "base_id":       base_id,
                "name":          name,
                "email":         email,
                "source":        source,
                "latest_sub_id": sub_id,
                "resume_json":   resume_dict,   # always reflects the latest version
                "updated_at":    now,
            },
            "$push": {
                "resumes": {
                    "sub_id":    sub_id,
                    "chroma_id": chroma_id,
                    "filename":  filename or "",
                    "added_at":  now,
                }
            },
            "$setOnInsert": {"added_at": now},
        },
        upsert=True,
    )


def save_candidate(candidate_id: str, resume_dict: dict, source: str = "seeded") -> None:
    """Simple upsert used by seed_candidates.py (no versioning needed for seeds)."""
    db = get_db()
    now = datetime.now(timezone.utc)
    email = ((resume_dict.get("contact_info") or {}).get("email") or "").strip().lower()
    name  = (resume_dict.get("contact_info") or {}).get("name") or candidate_id
    doc_key = email if (email and "@" in email) else candidate_id

    db.candidates.update_one(
        {"_id": doc_key},
        {
            "$set": {
                "base_id":       candidate_id,
                "name":          name,
                "email":         email,
                "source":        source,
                "latest_sub_id": 1,
                "resume_json":   resume_dict,
                "updated_at":    now,
            },
            "$push": {
                "resumes": {
                    "sub_id":    1,
                    "chroma_id": candidate_id,
                    "filename":  "seeded",
                    "added_at":  now,
                }
            },
            "$setOnInsert": {"added_at": now},
        },
        upsert=True,
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
        "candidates_evaluated": len(job_doc.get("rankings", [])),
        "rankings":             job_doc.get("rankings", []),
        "critique_summary":     (job_doc.get("critique") or {}).get("summary", ""),
        "run_at":               datetime.now(timezone.utc),
    })
    return str(result.inserted_id)


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
