"""
Example usage of the Resume Screening Assistant
"""

import os
import json
from dotenv import load_dotenv
from resume_parser import ResumeParser
from openrouter_client import OpenRouterClient
from agentic_pipeline import ScreeningPipeline, ScreeningQuery

# Load environment variables
load_dotenv()

# Example job description
JOB_DESCRIPTION = """
Senior Software Engineer - Full Stack

We are looking for an experienced Full Stack Engineer with:
- 5+ years of software development experience
- Strong proficiency in Python and JavaScript/TypeScript
- Experience with React or Vue.js
- Backend experience with Django, FastAPI, or similar frameworks
- Database design and SQL/NoSQL experience
- Experience with Docker and Kubernetes
- Cloud platform experience (AWS, GCP, or Azure)
- Strong problem-solving and communication skills

Responsibilities:
- Design and implement scalable backend services
- Build responsive frontend interfaces
- Mentor junior developers
- Participate in code reviews
- Contribute to architectural decisions
"""

EVALUATION_CRITERIA = {
    "required_skills": ["Python", "JavaScript", "React", "Docker", "AWS"],
    "minimum_experience_years": 5,
    "minimum_education": "Bachelor's degree in Computer Science or related field",
    "nice_to_have": ["Kubernetes", "ML/AI experience", "Open source contributions"]
}


def example_parse_resume():
    """Example: Parse a single resume"""
    print("=" * 60)
    print("EXAMPLE 1: Parse Resume from File")
    print("=" * 60)

    parser = ResumeParser()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    llm_client = OpenRouterClient(api_key)

    # Example: parse from text (simulating a resume)
    sample_resume_text = """
    John Doe
    Email: john.doe@example.com
    Phone: (555) 123-4567
    LinkedIn: linkedin.com/in/johndoe

    Professional Summary:
    Full-stack engineer with 7 years of experience building scalable web applications.
    Expert in Python, React, and cloud technologies.

    Skills:
    - Languages: Python, JavaScript, TypeScript, SQL
    - Frontend: React, Vue.js, HTML5, CSS3
    - Backend: Django, FastAPI, Flask, Node.js
    - DevOps: Docker, Kubernetes, AWS, CI/CD
    - Databases: PostgreSQL, MongoDB, Redis
    - Other: Git, REST APIs, GraphQL, Microservices

    Experience:
    Senior Software Engineer, Tech Corp (2021-Present)
    - Led development of microservices architecture serving 1M+ users
    - Implemented React-based dashboard used by 50K+ customers
    - Mentored 3 junior engineers
    - Technologies: Python, FastAPI, React, PostgreSQL, Docker, Kubernetes, AWS

    Software Engineer, StartupXYZ (2018-2021)
    - Built full-stack features from design to deployment
    - Optimized database queries reducing response time by 40%
    - Implemented CI/CD pipeline using GitHub Actions
    - Technologies: Django, Vue.js, PostgreSQL, Docker, AWS

    Junior Developer, WebSolutions (2015-2018)
    - Developed backend REST APIs using Django
    - Created responsive web interfaces with HTML, CSS, JavaScript
    - Participated in Agile development process
    - Technologies: Python, JavaScript, MySQL

    Education:
    Bachelor of Technology in Computer Science
    Indian Institute of Technology, Bombay (2015)
    GPA: 3.8/4.0

    Certifications:
    - AWS Solutions Architect Associate
    - Kubernetes for Developers
    """

    try:
        resume = llm_client.extract_resume_from_text(sample_resume_text)
        print("\n✓ Resume parsed successfully!")
        print(f"Name: {resume.contact_info.name}")
        print(f"Email: {resume.contact_info.email}")
        print(f"Skills: {[s.category for s in resume.skills]}")
        print(f"Experience entries: {len(resume.experience)}")
        print(f"Education entries: {len(resume.education)}")
        print(f"Certifications: {len(resume.certifications)}")

        # Save to JSON
        with open("parsed_resume.json", "w") as f:
            json.dump(resume.dict(), f, indent=2)
        print("\n✓ Saved to parsed_resume.json")

        return resume

    except Exception as e:
        print(f"✗ Error: {e}")


def example_screen_candidates():
    """Example: Screen multiple candidates using agentic pipeline"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Screen Candidates with Agentic Pipeline")
    print("=" * 60)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    # Initialize pipeline
    pipeline = ScreeningPipeline(api_key)
    llm_client = OpenRouterClient(api_key)

    # Sample candidate resumes
    sample_candidates = [
        {
            "id": "candidate_1",
            "text": """
            John Doe
            Email: john@example.com
            Phone: (555) 123-4567

            Professional Summary:
            Full-stack engineer with 7 years of experience.

            Skills: Python, JavaScript, React, FastAPI, Docker, Kubernetes, AWS, PostgreSQL

            Experience:
            Senior Software Engineer, Tech Corp (2021-Present) - 3 years
            - Led microservices development
            - Python, FastAPI, React, AWS, Docker, Kubernetes

            Software Engineer, StartupXYZ (2018-2021) - 3 years
            - Built full-stack features
            - Django, Vue.js, PostgreSQL, Docker

            Education:
            Bachelor of Technology in Computer Science, IIT Bombay (2015)
            """
        },
        {
            "id": "candidate_2",
            "text": """
            Jane Smith
            Email: jane@example.com

            Professional Summary:
            Backend engineer with 4 years of experience.

            Skills: Python, JavaScript, FastAPI, Flask, PostgreSQL, MongoDB

            Experience:
            Backend Engineer, CloudStartup (2022-Present) - 2 years
            - REST API development
            - Python, FastAPI, PostgreSQL

            Junior Developer, WebCorp (2020-2022) - 2 years
            - Basic backend development
            - Python, Flask

            Education:
            Bachelor of Computer Science, State University (2020)
            """
        },
        {
            "id": "candidate_3",
            "text": """
            Mike Johnson
            Email: mike@example.com

            Professional Summary:
            Full-stack engineer with 8 years of experience in large-scale systems.

            Skills: Python, Go, Rust, JavaScript, React, Vue, Docker, Kubernetes, AWS, GCP, Azure, PostgreSQL, MongoDB, Redis

            Experience:
            Principal Engineer, MegaCorp (2022-Present) - 2 years
            - Architected distributed systems
            - Python, Go, Kubernetes, AWS, GCP

            Senior Software Engineer, TechGiant (2018-2022) - 4 years
            - Led team of 5 engineers
            - Microservices, Docker, Kubernetes, AWS

            Software Engineer, StartupABC (2016-2018) - 2 years
            - Full-stack development
            - Python, JavaScript, React, Docker

            Education:
            Master of Computer Science, Carnegie Mellon University (2016)
            Bachelor of Computer Science, UC Berkeley (2014)

            Certifications: AWS Solutions Architect, Kubernetes Administrator
            """
        }
    ]

    # Parse and add candidates to pipeline
    print("\nAdding candidates to pipeline...")
    for candidate in sample_candidates:
        try:
            resume = llm_client.extract_resume_from_text(candidate["text"])
            pipeline.add_candidate(resume, candidate["id"])
            name = resume.contact_info.name if resume.contact_info else candidate["id"]
            print(f"  ✓ Added {name}")
        except Exception as e:
            print(f"  ✗ Error adding {candidate['id']}: {e}")

    # Create screening query
    query = ScreeningQuery(
        job_description=JOB_DESCRIPTION,
        required_skills=EVALUATION_CRITERIA["required_skills"],
        min_experience_years=EVALUATION_CRITERIA["minimum_experience_years"],
        required_education=EVALUATION_CRITERIA["minimum_education"],
        evaluation_criteria=EVALUATION_CRITERIA
    )

    # Run screening pipeline
    print("\nRunning screening pipeline...")
    state = pipeline.screen_candidates(query)

    print(f"\n✓ Screening complete!")
    print(f"  Total candidates: {state.plan.get('top_candidate_count', len(state.candidates))}")
    print(f"  Retrieved: {len(state.retrieved_candidates)}")
    print(f"  Evaluated: {len(state.evaluations)}")

    # Display top candidates
    print("\n" + "=" * 60)
    print("TOP CANDIDATES:")
    print("=" * 60)

    for ranking in state.final_rankings[:5]:
        print(f"\nRank #{ranking['rank']}")
        print(f"  Candidate ID: {ranking['candidate_id']}")
        print(f"  Match Score: {ranking['match_score']}/100")
        print(f"  Recommendation: {ranking['overall_recommendation']}")
        print(f"  Reasoning: {ranking['reasoning'][:200]}...")

    # Save results
    results = {
        "plan": state.plan,
        "total_candidates": len(state.candidates),
        "evaluations": state.evaluations,
        "final_rankings": state.final_rankings,
        "critique": state.critique_feedback
    }

    with open("screening_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved screening results to screening_results.json")


def example_evaluate_single():
    """Example: Evaluate a single candidate"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Evaluate Single Candidate")
    print("=" * 60)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    llm_client = OpenRouterClient(api_key)

    # Sample resume
    sample_resume = """
    Jane Smith
    Email: jane@example.com

    Professional Summary:
    Backend engineer with 4 years of experience in Python and JavaScript.

    Skills: Python, JavaScript, FastAPI, Flask, PostgreSQL, MongoDB, Docker

    Experience:
    Backend Engineer, CloudStartup (2022-Present)
    - REST API development using FastAPI
    - Database optimization and design
    - Docker containerization

    Junior Developer, WebCorp (2020-2022)
    - Python and JavaScript development
    - SQL database design

    Education:
    Bachelor of Computer Science, State University (2020)
    """

    try:
        resume = llm_client.extract_resume_from_text(sample_resume)
        evaluation = llm_client.evaluate_candidate(
            resume,
            JOB_DESCRIPTION,
            EVALUATION_CRITERIA
        )

        print(f"\n✓ Evaluation complete!")
        print(f"  Match Score: {evaluation['match_score']}/100")
        print(f"  Recommendation: {evaluation['overall_recommendation']}")
        print(f"  Strengths: {', '.join(evaluation['strengths'][:3])}")
        print(f"  Weaknesses: {', '.join(evaluation['weaknesses'][:3])}")

        with open("single_evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)
        print("\n✓ Saved evaluation to single_evaluation.json")

    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESUME SCREENING ASSISTANT - EXAMPLES")
    print("=" * 60)

    # Run examples
    try:
        example_parse_resume()
        example_screen_candidates()
        example_evaluate_single()

        print("\n" + "=" * 60)
        print("✓ ALL EXAMPLES COMPLETED")
        print("=" * 60)
        print("\nCheck the generated JSON files for detailed results:")
        print("  - parsed_resume.json")
        print("  - screening_results.json")
        print("  - single_evaluation.json")

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
