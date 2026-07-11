import requests
import json
import os
from typing import Optional, Dict, List, Any
from schemas import Resume


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "deepseek/deepseek-chat"  # DeepSeek v4 Flash equivalent

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> str:
        """Call OpenRouter LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Resume Screening Assistant"
        }

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")

    def extract_resume_from_text(self, text: str) -> Resume:
        """Use LLM to extract structured resume data from raw text"""
        extraction_prompt = f"""
You are an expert resume parser. Extract ALL information from the following resume text and return it as valid JSON.
Return the data in this exact JSON structure:
{{
    "contact_info": {{
        "name": string,
        "email": string,
        "phone": string,
        "address": string,
        "city": string,
        "state": string,
        "country": string,
        "zip_code": string,
        "linkedin": string,
        "github": string,
        "portfolio": string,
        "website": string
    }},
    "professional_summary": string,
    "skills": [
        {{"category": string, "skills": [string]}}
    ],
    "experience": [
        {{
            "company": string,
            "position": string,
            "start_date": string,
            "end_date": string,
            "employment_type": string,
            "location": string,
            "description": string,
            "achievements": [string],
            "technologies": [string]
        }}
    ],
    "education": [
        {{
            "institution": string,
            "degree": string,
            "field_of_study": string,
            "start_date": string,
            "end_date": string,
            "gpa": string,
            "activities": string,
            "description": string
        }}
    ],
    "certifications": [
        {{
            "title": string,
            "issuer": string,
            "date_obtained": string,
            "expiration_date": string,
            "credential_url": string
        }}
    ],
    "projects": [
        {{
            "title": string,
            "description": string,
            "technologies": [string],
            "url": string,
            "start_date": string,
            "end_date": string
        }}
    ],
    "languages": [
        {{"language": string, "proficiency": string}}
    ],
    "publications": [string],
    "volunteer_experience": [string],
    "awards_recognition": [string]
}}

Resume text:
{text}

Return ONLY valid JSON, no markdown or extra text.
"""

        messages = [
            {"role": "user", "content": extraction_prompt}
        ]

        try:
            response = self.call_llm(messages, temperature=0.3)
            # Clean up response
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()

            data = json.loads(json_str)
            return self._dict_to_resume(data)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM response as JSON: {str(e)}")

    def _dict_to_resume(self, data: dict) -> Resume:
        """Convert raw dict from LLM into nested Resume dataclass"""
        from schemas import ContactInfo, Education, Experience, Skill, Certification, Project, Language

        ci = data.get("contact_info") or {}
        contact_info = ContactInfo(**{k: v for k, v in ci.items() if k in ContactInfo.__dataclass_fields__}) if ci else ContactInfo()

        skills = [
            Skill(category=s.get("category"), skills=s.get("skills", []))
            for s in (data.get("skills") or [])
        ]
        experience = [
            Experience(**{k: v for k, v in e.items() if k in Experience.__dataclass_fields__})
            for e in (data.get("experience") or [])
        ]
        education = [
            Education(**{k: v for k, v in e.items() if k in Education.__dataclass_fields__})
            for e in (data.get("education") or [])
        ]
        certifications = [
            Certification(**{k: v for k, v in c.items() if k in Certification.__dataclass_fields__})
            for c in (data.get("certifications") or [])
        ]
        projects = [
            Project(**{k: v for k, v in p.items() if k in Project.__dataclass_fields__})
            for p in (data.get("projects") or [])
        ]
        languages = [
            Language(language=l.get("language"), proficiency=l.get("proficiency"))
            for l in (data.get("languages") or [])
        ]

        return Resume(
            contact_info=contact_info,
            professional_summary=data.get("professional_summary"),
            skills=skills,
            experience=experience,
            education=education,
            certifications=certifications,
            projects=projects,
            languages=languages,
            publications=data.get("publications") or [],
            volunteer_experience=data.get("volunteer_experience") or [],
            awards_recognition=data.get("awards_recognition") or [],
            metadata=data.get("metadata") or {}
        )

    def evaluate_candidate(
        self,
        resume: Resume,
        job_description: str,
        evaluation_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate candidate against job description"""
        evaluation_prompt = f"""
You are an expert recruiter. Evaluate the following candidate resume against the job description.
Provide a structured evaluation.

Resume JSON:
{json.dumps(resume.dict(), indent=2)}

Job Description:
{job_description}

Additional Evaluation Criteria:
{json.dumps(evaluation_criteria or {}, indent=2)}

Provide your evaluation as JSON with this structure:
{{
    "match_score": float (0-100),
    "strengths": [string],
    "weaknesses": [string],
    "skill_match": {{"required_skills": [string], "matched_skills": [string], "missing_skills": [string]}},
    "experience_match": string,
    "education_match": string,
    "overall_recommendation": "STRONG_MATCH" | "MATCH" | "WEAK_MATCH" | "NOT_QUALIFIED",
    "reasoning": string,
    "questions_to_ask": [string]
}}

Return ONLY valid JSON.
"""

        messages = [
            {"role": "user", "content": evaluation_prompt}
        ]

        response = self.call_llm(messages, temperature=0.3)
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()

        return json.loads(json_str)
