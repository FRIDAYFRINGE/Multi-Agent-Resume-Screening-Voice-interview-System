"""
Interview Simulator — LangGraph state machine.

The graph owns the interview's control flow.  It asks a question, waits for an
answer, scores it, and *decides* whether to probe further or move on.  That
decision is a conditional edge with a cycle back to ask_question, so the path
through the graph differs per candidate: a shallow answer earns a follow-up, a
strong one does not.

I/O lives in the driver, not in the nodes.  listen_node calls interrupt(), so
the graph pauses and whoever is driving supplies the transcribed answer via
Command(resume=text).  That is what lets one graph serve both the CLI (mic +
local Whisper) and the web app (browser audio + Groq/Whisper) without either
side re-implementing the follow-up logic.
"""
import json
import operator
import uuid
from typing import Dict, List, Any, Optional, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from openrouter_client import OpenRouterClient, MODEL_V3


# ─── State ────────────────────────────────────────────────────────────────────

class InterviewState(TypedDict):
    candidate_id: str
    candidate_name: str
    job_description: str
    question_queue: List[Dict[str, Any]]   # ordered flat list of questions
    current_idx: int                        # which question we're on
    current_question: Optional[Dict]        # the active question dict
    follow_up_count: int                    # follow-ups used on current Q
    max_follow_ups: int                     # max follow-ups per question (default 1)
    last_answer: str                        # most recent transcribed answer
    transcript: Annotated[List[Dict], operator.add]  # full Q&A log
    answer_scores: Annotated[List[Dict], operator.add]  # per-answer scores
    is_complete: bool
    final_report: Optional[Dict]


# Answers shorter than this are treated as silence and scored 0 without an LLM
# call.  Browser audio routinely produces one- or two-word noise transcriptions.
_MIN_ANSWER_WORDS = 4


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    from openrouter_client import _extract_json
    return _extract_json(text)


def _flatten_questions(interview_questions: Dict[str, Any]) -> List[Dict]:
    """
    Flatten the interview_questions dict produced by the screening pipeline
    into an ordered list with category labels.

    Order: technical_depth → gap_probing → project_specific → system_design → behavioral
    """
    order = ["technical_depth", "gap_probing", "project_specific", "system_design", "behavioral"]
    flat = []
    for category in order:
        items = interview_questions.get(category, [])
        for item in items:
            flat.append({
                "category": category,
                "question": item.get("question", ""),
                "what_to_look_for": item.get("what_to_look_for", ""),
                "follow_up": item.get("follow_up", ""),
                "gap": item.get("gap", ""),
                "project": item.get("project", ""),
            })
    return flat


def interrupt_payload(result: Optional[dict]) -> Optional[dict]:
    """Return the pending interrupt's payload, or None if the graph finished.

    Drivers use this to tell "the graph is waiting for an answer" apart from
    "the graph reached END", which is the only branch they need to care about.
    """
    pending = (result or {}).get("__interrupt__")
    if not pending:
        return None
    first = pending[0] if isinstance(pending, (list, tuple)) else pending
    return getattr(first, "value", first)


def initial_state(
    candidate_id: str,
    candidate_name: str,
    job_description: str,
    question_queue: List[Dict[str, Any]],
    max_follow_ups: int = 1,
) -> InterviewState:
    """Build a fresh graph state. Shared by the CLI and web drivers."""
    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "job_description": job_description,
        "question_queue": list(question_queue),
        "current_idx": 0,
        "current_question": None,
        "follow_up_count": 0,
        "max_follow_ups": max_follow_ups,
        "last_answer": "",
        "transcript": [],
        "answer_scores": [],
        "is_complete": False,
        "final_report": None,
    }


def thread_config(thread_id: str, recursion_limit: int = 200) -> dict:
    """Checkpointer config for one interview. thread_id scopes the saved state."""
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": recursion_limit}


# ─── Agent Nodes ──────────────────────────────────────────────────────────────

class InterviewNodes:
    """Pure-logic nodes. No printing, no audio — the driver renders."""

    def __init__(self, llm: OpenRouterClient):
        self.llm = llm

    # ── Load Questions ────────────────────────────────────────────────────────
    def load_questions_node(self, state: InterviewState) -> dict:
        """Entry point.

        The cursor is carried through rather than zeroed, so a driver can seed
        partial progress and have the interview resume where it left off.
        """
        return {"current_idx": state.get("current_idx", 0),
                "follow_up_count": state.get("follow_up_count", 0),
                "is_complete": False}

    # ── Ask Question ──────────────────────────────────────────────────────────
    def ask_question_node(self, state: InterviewState) -> dict:
        """Select the active question. Delivery is the driver's job."""
        idx = state["current_idx"]
        queue = state["question_queue"]

        if idx >= len(queue):
            return {"is_complete": True, "current_question": None}

        return {"current_question": queue[idx]}

    # ── Listen ────────────────────────────────────────────────────────────────
    def listen_node(self, state: InterviewState) -> dict:
        """Pause the graph until the driver supplies a transcribed answer.

        The interrupt payload carries everything a driver needs to present the
        question, so neither the CLI nor the web app has to reach into graph
        state to find out what was asked.
        """
        q = state.get("current_question") or {}
        answer = interrupt({
            "type": "question",
            "idx": state["current_idx"],
            "total": len(state["question_queue"]),
            "category": q.get("category", ""),
            "question": q.get("question", ""),
            "gap": q.get("gap", ""),
            "what_to_look_for": q.get("what_to_look_for", ""),
            "is_follow_up": state.get("follow_up_count", 0) > 0,
        })

        # Accept either a plain string or {"answer": ...} so a driver can pass
        # extra metadata later without breaking this contract.
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        return {"last_answer": (answer or "").strip()}

    # ── Evaluate Answer ───────────────────────────────────────────────────────
    def evaluate_node(self, state: InterviewState) -> dict:
        """LLM scores the answer against the expected signal."""
        q = state["current_question"] or {}
        answer = state["last_answer"]

        # Silence guard — no speech means no LLM call.  Lives here rather than
        # in a driver so the CLI and the web app cannot disagree about what
        # counts as an empty answer.
        if not answer or len(answer.split()) < _MIN_ANSWER_WORDS:
            evaluation = {
                "score": 0, "depth": "shallow",
                "hits": [], "misses": ["No answer given"],
                "needs_follow_up": False, "follow_up_reason": "",
                "brief_feedback": "No speech detected or answer too short.",
            }
        else:
            prompt = f"""Evaluate this interview answer.
Question: {q.get('question', '')}
What to look for: {q.get('what_to_look_for', '')}
Candidate's answer (raw speech-to-text — may have minor accent/STT errors like wrong word endings or near-homophones; infer the intended technical term from context):
{answer}

Return ONLY valid JSON:
{{"score": <int 1-10>, "depth": "<shallow|adequate|deep>",
  "hits": ["<covered>"], "misses": ["<missed>"],
  "needs_follow_up": <true|false>, "follow_up_reason": "<why>",
  "brief_feedback": "<1 sentence>"}}

Score guide: 1-3=poor, 4-6=adequate, 7-8=good, 9-10=excellent.
needs_follow_up=true only if the answer was shallow or missed a critical signal."""

            try:
                resp = self.llm.call_llm(
                    [{"role": "user", "content": prompt}], model=MODEL_V3, temperature=0.1
                )
                evaluation = _parse_json(resp)
            except Exception as e:
                evaluation = {
                    "score": 0, "depth": "error", "hits": [], "misses": [],
                    "needs_follow_up": False, "follow_up_reason": "", "brief_feedback": str(e),
                }

        entry = {
            "question_idx": state["current_idx"],
            "category": q.get("category", ""),
            "question": q.get("question", ""),
            "answer": answer,
            "evaluation": evaluation,
            "is_follow_up": state.get("follow_up_count", 0) > 0,
        }
        score_entry = {
            "question_idx": state["current_idx"],
            "category": q.get("category", ""),
            "score": evaluation.get("score", 0),
            "depth": evaluation.get("depth", ""),
        }

        return {"transcript": [entry], "answer_scores": [score_entry]}

    # ── Decide ────────────────────────────────────────────────────────────────
    def decide_node(self, state: InterviewState) -> dict:
        """
        Decide what happens next:
        - follow_up  : ask a follow-up for the current question
        - next       : move to the next question
        - done       : all questions exhausted → complete
        """
        if state.get("is_complete"):
            return {"is_complete": True}

        transcript = state.get("transcript", [])
        if not transcript:
            return {"current_idx": state["current_idx"] + 1, "follow_up_count": 0}

        last_eval = transcript[-1]["evaluation"]
        needs_follow_up = last_eval.get("needs_follow_up", False)
        follow_up_count = state["follow_up_count"]
        max_follow_ups = state["max_follow_ups"]
        current_q = state.get("current_question") or {}

        if needs_follow_up and follow_up_count < max_follow_ups and current_q.get("follow_up"):
            # Splice the follow-up in right after the current position.
            follow_up_q = {
                **current_q,
                "question": current_q["follow_up"],
                "what_to_look_for": current_q.get("what_to_look_for", ""),
                "follow_up": "",  # no further follow-up on a follow-up
            }
            idx = state["current_idx"]
            new_queue = list(state["question_queue"])
            new_queue.insert(idx + 1, follow_up_q)

            return {
                "question_queue": new_queue,
                "current_idx": idx + 1,
                "follow_up_count": follow_up_count + 1,
            }

        next_idx = state["current_idx"] + 1
        if next_idx >= len(state["question_queue"]):
            return {"is_complete": True}
        return {"current_idx": next_idx, "follow_up_count": 0}

    # ── Generate Report ───────────────────────────────────────────────────────
    def report_node(self, state: InterviewState) -> dict:
        """LLM generates a final interview report from the full transcript."""
        transcript = state.get("transcript", [])
        scores = state.get("answer_scores", [])
        avg_score = sum(s["score"] for s in scores) / len(scores) if scores else 0

        prompt = f"""You are a senior hiring manager. Write a final interview assessment report.

Candidate: {state['candidate_name']}
Job Description: {state['job_description'][:400]}

Full interview transcript:
{json.dumps(transcript, indent=2)}

Average score: {avg_score:.1f}/10

Return ONLY valid JSON:
{{
  "overall_score": <float 1-10>,
  "verdict": "<HIRE|STRONG_HIRE|HOLD|REJECT>",
  "summary": "<2-3 sentence overall impression>",
  "category_scores": {{
    "technical_depth": <float 1-10>,
    "gap_probing": <float 1-10>,
    "project_specific": <float 1-10>,
    "system_design": <float 1-10>,
    "behavioral": <float 1-10>
  }},
  "top_strengths": ["<strength observed during interview>"],
  "key_concerns": ["<concern observed during interview>"],
  "recommendation": "<detailed hiring recommendation paragraph>"
}}"""

        try:
            resp = self.llm.call_llm(
                [{"role": "user", "content": prompt}], model=MODEL_V3,
                temperature=0.1, max_tokens=1024,
            )
            report = _parse_json(resp)
        except Exception as e:
            report = {"error": str(e), "overall_score": avg_score, "verdict": "ERROR"}

        report["candidate_id"] = state["candidate_id"]
        report["candidate_name"] = state["candidate_name"]
        report["avg_raw_score"] = round(avg_score, 2)
        report["total_questions"] = len(transcript)

        return {"final_report": report, "is_complete": True}


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_ask(state: InterviewState) -> str:
    """Nothing left to ask → report. Otherwise wait for an answer."""
    return "report" if state.get("is_complete") else "listen"


def route_after_decide(state: InterviewState) -> str:
    """The agentic branch: probe further, or wrap up."""
    return "report" if state.get("is_complete") else "ask_question"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_interview_graph(llm: OpenRouterClient, checkpointer=None):
    """Compile the interview graph.

    A checkpointer is required for interrupt/resume to work at all, so one is
    supplied by default rather than left to each caller to remember.
    """
    nodes = InterviewNodes(llm)

    graph = StateGraph(InterviewState)
    graph.add_node("load_questions", nodes.load_questions_node)
    graph.add_node("ask_question",   nodes.ask_question_node)
    graph.add_node("listen",         nodes.listen_node)
    graph.add_node("evaluate",       nodes.evaluate_node)
    graph.add_node("decide",         nodes.decide_node)
    graph.add_node("report",         nodes.report_node)

    graph.set_entry_point("load_questions")
    graph.add_edge("load_questions", "ask_question")

    graph.add_conditional_edges(
        "ask_question",
        route_after_ask,
        {"listen": "listen", "report": "report"},
    )

    graph.add_edge("listen",   "evaluate")
    graph.add_edge("evaluate", "decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"ask_question": "ask_question", "report": "report"},
    )

    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# ─── Facade (CLI driver) ──────────────────────────────────────────────────────

class InterviewSimulator:
    """Drives the interview graph from a terminal: TTS out, microphone in."""

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        whisper_model: str = "base",
        tts_rate: int = 165,
        max_follow_ups: int = 1
    ):
        from openrouter_client import OpenRouterClient
        from voice_io import TextToSpeech, SpeechToText
        self.llm = OpenRouterClient(openrouter_api_key)
        self.tts = TextToSpeech(rate=tts_rate)
        self.stt = SpeechToText(model_size=whisper_model)
        self.max_follow_ups = max_follow_ups
        self._graph = build_interview_graph(self.llm)

    def run(
        self,
        candidate_id: str,
        candidate_name: str,
        job_description: str,
        interview_questions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a full live interview session.

        interview_questions: the dict produced by the screening pipeline's
        interview_questions_node for this candidate.
        """
        queue = _flatten_questions(interview_questions)

        if not queue:
            raise ValueError("No interview questions provided for this candidate")

        config = thread_config(f"cli-{candidate_id}-{uuid.uuid4().hex[:8]}")
        state = initial_state(
            candidate_id, candidate_name, job_description, queue, self.max_follow_ups
        )

        intro = (
            f"Hello {candidate_name}. Welcome to your technical interview. "
            f"I will be asking you {len(queue)} questions across different areas. "
            f"Take your time to answer thoroughly. Let's begin."
        )
        print(f"\n{'='*60}")
        print(f"  INTERVIEW — {candidate_name.upper()}")
        print(f"  {len(queue)} questions | max {self.max_follow_ups} follow-up per question")
        print(f"{'='*60}\n")
        print(f"[INTERVIEWER] {intro}\n")
        self.tts.speak(intro)

        result = self._graph.invoke(state, config)

        while True:
            payload = interrupt_payload(result)
            if payload is None:
                break  # graph reached END — report is ready

            cat = (payload.get("category") or "").replace("_", " ").title()
            marker = " (follow-up)" if payload.get("is_follow_up") else ""
            print(f"\n[Q{payload['idx'] + 1}/{payload['total']}] [{cat}]{marker}")
            print(f"[INTERVIEWER] {payload['question']}\n")
            self.tts.speak(payload["question"])

            print("[CANDIDATE] (listening... press Enter when done speaking)")
            answer = self.stt.listen(duration=120)
            print(f"[CANDIDATE] {answer}\n")

            result = self._graph.invoke(Command(resume=answer), config)

            transcript = result.get("transcript") or []
            if transcript:
                ev = transcript[-1]["evaluation"]
                print(f"[EVAL] Score: {ev.get('score', 0)}/10 | Depth: {ev.get('depth', '')}")
                if ev.get("needs_follow_up") and ev.get("follow_up_reason"):
                    print(f"[DECISION] Follow-up needed: {ev['follow_up_reason']}")

        closing = ("Thank you for your time. That concludes the interview. "
                   "We will be in touch with feedback soon.")
        print(f"\n[INTERVIEWER] {closing}")
        self.tts.speak(closing)

        report = result.get("final_report") or {}
        print(f"\n{'='*60}")
        print(f"  INTERVIEW REPORT — {candidate_name.upper()}")
        print(f"{'='*60}")
        print(f"  Overall Score : {report.get('overall_score', '?')}/10")
        print(f"  Verdict       : {report.get('verdict', '?')}")
        print(f"  Summary       : {report.get('summary', '')}")
        print(f"\n  Strengths:")
        for s in report.get("top_strengths", []):
            print(f"    + {s}")
        print(f"\n  Concerns:")
        for c in report.get("key_concerns", []):
            print(f"    - {c}")
        print(f"{'='*60}\n")

        return report
