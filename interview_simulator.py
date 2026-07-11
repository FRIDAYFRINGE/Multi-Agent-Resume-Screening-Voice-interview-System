"""
Interview Simulator — LangGraph state machine
Speaks questions via TTS, records answers via STT (Whisper),
evaluates responses with LLM, decides follow-up or next question,
produces a final scored report.
"""
import json
import sys
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from openrouter_client import OpenRouterClient
from voice_io import TextToSpeech, SpeechToText


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


# ─── Agent Nodes ──────────────────────────────────────────────────────────────

class InterviewNodes:
    def __init__(self, llm: OpenRouterClient, tts: TextToSpeech, stt: SpeechToText):
        self.llm = llm
        self.tts = tts
        self.stt = stt

    # ── Load Questions ────────────────────────────────────────────────────────
    def load_questions_node(self, state: InterviewState) -> dict:
        """Already loaded — just announce the start and set index"""
        name = state["candidate_name"]
        total = len(state["question_queue"])

        intro = (
            f"Hello {name}. Welcome to your technical interview. "
            f"I will be asking you {total} questions across different areas. "
            f"Take your time to answer thoroughly. Let's begin."
        )
        print(f"\n{'='*60}")
        print(f"  INTERVIEW — {name.upper()}")
        print(f"  {total} questions | max {state['max_follow_ups']} follow-up per question")
        print(f"{'='*60}\n")
        print(f"[INTERVIEWER] {intro}\n")
        self.tts.speak(intro)

        return {"current_idx": 0, "follow_up_count": 0, "is_complete": False}

    # ── Ask Question ──────────────────────────────────────────────────────────
    def ask_question_node(self, state: InterviewState) -> dict:
        """Speak the current question aloud"""
        idx = state["current_idx"]
        queue = state["question_queue"]

        if idx >= len(queue):
            return {"is_complete": True, "current_question": None}

        q = queue[idx]
        cat = q["category"].replace("_", " ").title()
        text = q["question"]

        print(f"\n[Q{idx+1}/{len(queue)}] [{cat}]")
        print(f"[INTERVIEWER] {text}\n")
        self.tts.speak(text)

        return {"current_question": q}

    # ── Listen ────────────────────────────────────────────────────────────────
    def listen_node(self, state: InterviewState) -> dict:
        """Record candidate answer and transcribe via Whisper"""
        print("[CANDIDATE] (listening... press Enter when done speaking)")
        answer = self.stt.listen(duration=120)

        if not answer.strip():
            answer = "[No answer detected]"

        print(f"[CANDIDATE] {answer}\n")
        return {"last_answer": answer}

    # ── Evaluate Answer ───────────────────────────────────────────────────────
    def evaluate_node(self, state: InterviewState) -> dict:
        """LLM scores the answer against expected signal"""
        q = state["current_question"]
        answer = state["last_answer"]

        prompt = f"""You are a senior technical interviewer evaluating a candidate's answer.

Question: {q['question']}

What to look for: {q['what_to_look_for']}

Candidate's answer: {answer}

Evaluate the answer and return ONLY valid JSON:
{{
  "score": <int 1-10>,
  "depth": "<shallow|adequate|deep>",
  "hits": ["<signal the candidate covered>"],
  "misses": ["<signal the candidate missed>"],
  "needs_follow_up": <true|false>,
  "follow_up_reason": "<why a follow-up is needed, or empty string>",
  "brief_feedback": "<1 sentence internal note for the report>"
}}

Score guide: 1-3=poor, 4-6=adequate, 7-8=good, 9-10=excellent.
needs_follow_up=true only if the answer was shallow or missed a critical signal."""

        try:
            resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.1)
            evaluation = _parse_json(resp)
        except Exception as e:
            evaluation = {
                "score": 0, "depth": "error", "hits": [], "misses": [],
                "needs_follow_up": False, "follow_up_reason": "", "brief_feedback": str(e)
            }

        # Log to transcript
        entry = {
            "question_idx": state["current_idx"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "evaluation": evaluation,
            "is_follow_up": state["follow_up_count"] > 0
        }

        score_entry = {
            "question_idx": state["current_idx"],
            "category": q["category"],
            "score": evaluation.get("score", 0),
            "depth": evaluation.get("depth", ""),
        }

        score = evaluation.get("score", 0)
        depth = evaluation.get("depth", "")
        print(f"[EVAL] Score: {score}/10 | Depth: {depth}")

        return {
            "transcript": [entry],
            "answer_scores": [score_entry]
        }

    # ── Decide ────────────────────────────────────────────────────────────────
    def decide_node(self, state: InterviewState) -> dict:
        """
        Decide what happens next:
        - follow_up  : ask follow-up for current question
        - next       : move to next question
        - done       : all questions exhausted → complete
        """
        if state.get("is_complete"):
            return {"is_complete": True}

        # Get last evaluation
        transcript = state.get("transcript", [])
        if not transcript:
            return {"current_idx": state["current_idx"] + 1, "follow_up_count": 0}

        last_eval = transcript[-1]["evaluation"]
        needs_follow_up = last_eval.get("needs_follow_up", False)
        follow_up_count = state["follow_up_count"]
        max_follow_ups = state["max_follow_ups"]
        current_q = state["current_question"]

        if needs_follow_up and follow_up_count < max_follow_ups and current_q.get("follow_up"):
            # Inject follow-up into queue at current position
            follow_up_q = {
                **current_q,
                "question": current_q["follow_up"],
                "what_to_look_for": current_q["what_to_look_for"],
                "follow_up": "",  # no further follow-up on a follow-up
            }
            reason = last_eval.get("follow_up_reason", "")
            print(f"\n[DECISION] Follow-up needed: {reason}")

            # Splice follow-up into queue right after current position
            idx = state["current_idx"]
            new_queue = list(state["question_queue"])
            new_queue.insert(idx + 1, follow_up_q)

            return {
                "question_queue": new_queue,
                "current_idx": idx + 1,
                "follow_up_count": follow_up_count + 1
            }
        else:
            # Move to next question, reset follow-up count
            next_idx = state["current_idx"] + 1
            if next_idx >= len(state["question_queue"]):
                print("\n[DECISION] All questions complete.")
                return {"is_complete": True}
            else:
                print(f"\n[DECISION] Moving to question {next_idx + 1}.")
                return {"current_idx": next_idx, "follow_up_count": 0}

    # ── Generate Report ───────────────────────────────────────────────────────
    def report_node(self, state: InterviewState) -> dict:
        """LLM generates a final interview report from the full transcript"""
        closing = "Thank you for your time. That concludes the interview. We will be in touch with feedback soon."
        print(f"\n[INTERVIEWER] {closing}")
        self.tts.speak(closing)

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
            resp = self.llm.call_llm([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1024)
            report = _parse_json(resp)
        except Exception as e:
            report = {"error": str(e), "overall_score": avg_score, "verdict": "ERROR"}

        report["candidate_id"] = state["candidate_id"]
        report["candidate_name"] = state["candidate_name"]
        report["avg_raw_score"] = round(avg_score, 2)
        report["total_questions"] = len(transcript)

        # Print report summary
        print(f"\n{'='*60}")
        print(f"  INTERVIEW REPORT — {state['candidate_name'].upper()}")
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

        return {"final_report": report}


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_decide(state: InterviewState) -> str:
    if state.get("is_complete"):
        return "report"
    return "ask_question"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_interview_graph(llm: OpenRouterClient, tts: TextToSpeech, stt: SpeechToText):
    nodes = InterviewNodes(llm, tts, stt)

    graph = StateGraph(InterviewState)
    graph.add_node("load_questions", nodes.load_questions_node)
    graph.add_node("ask_question",   nodes.ask_question_node)
    graph.add_node("listen",         nodes.listen_node)
    graph.add_node("evaluate",       nodes.evaluate_node)
    graph.add_node("decide",         nodes.decide_node)
    graph.add_node("report",         nodes.report_node)

    graph.set_entry_point("load_questions")
    graph.add_edge("load_questions", "ask_question")
    graph.add_edge("ask_question",   "listen")
    graph.add_edge("listen",         "evaluate")
    graph.add_edge("evaluate",       "decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"ask_question": "ask_question", "report": "report"}
    )

    graph.add_edge("report", END)

    return graph.compile()


# ─── Facade ───────────────────────────────────────────────────────────────────

class InterviewSimulator:
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        whisper_model: str = "base",
        tts_rate: int = 165,
        max_follow_ups: int = 1
    ):
        from openrouter_client import OpenRouterClient
        self.llm = OpenRouterClient(openrouter_api_key)
        self.tts = TextToSpeech(rate=tts_rate)
        self.stt = SpeechToText(model_size=whisper_model)
        self.max_follow_ups = max_follow_ups
        self._graph = build_interview_graph(self.llm, self.tts, self.stt)

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

        initial: InterviewState = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "job_description": job_description,
            "question_queue": queue,
            "current_idx": 0,
            "current_question": None,
            "follow_up_count": 0,
            "max_follow_ups": self.max_follow_ups,
            "last_answer": "",
            "transcript": [],
            "answer_scores": [],
            "is_complete": False,
            "final_report": None,
        }

        final_state = self._graph.invoke(initial, {"recursion_limit": 200})
        return final_state.get("final_report", {})
