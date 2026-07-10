# Multi-Agent Resume Screening Assistant

An intelligent, agentic RAG system for automated resume screening using LangGraph, ChromaDB, and FastAPI. The system intelligently decomposes screening queries, retrieves relevant candidates through semantic search, and validates evaluations before generating final rankings.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Resume Input                              │
│            (PDF/DOCX/TXT) → PyMuPDF Extraction              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Resume Parser (LLM-Based)                       │
│         OpenRouter DeepSeek v4 Flash / Similar              │
│  Extracts: Contact, Skills, Experience, Education, etc.     │
│  Output: Structured JSON Resume                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vector Database                             │
│              ChromaDB - Semantic Search                      │
│        (Embeddings + Similarity Search)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐         ┌──────────────────┐
│  Planner Agent   │         │ Retrieval Agent  │
│ Decomposes Query │         │ Semantic Search  │
│ & Creates Plan   │         │  (ChromaDB)      │
└──────────────────┘         └──────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │    Evaluation Agent         │
        │ Score Candidates vs JD      │
        │ (Structured Extraction)     │
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │     Critique Agent          │
        │ Validate Evaluations        │
        │ Check Against JD            │
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │   Final Rankings & Report   │
        │   Sorted by Match Score     │
        └─────────────────────────────┘
```

## Features

- **Comprehensive Resume Parsing**: Extracts ALL fields including contact info, skills, experience, education, certifications, projects, languages, etc.
- **JSON Output**: Structured JSON format for easy integration
- **Multi-Format Support**: PDF, DOCX, TXT files
- **Semantic Search**: ChromaDB-powered vector search for intelligent candidate retrieval
- **Agentic Pipeline**: 
  - Planner: Decomposes screening queries
  - Retriever: Semantic search across candidates
  - Evaluator: LLM-based candidate evaluation
  - Critic: Validates evaluations against job requirements
- **REST API**: FastAPI-based REST endpoints for easy integration
- **Bulk Operations**: Upload and screen multiple candidates at once

## Components

### 1. `schemas.py`
Pydantic models for structured resume data:
- `ContactInfo`: Name, email, phone, social profiles
- `Education`: School, degree, field, dates, GPA
- `Experience`: Company, position, dates, responsibilities, technologies
- `Skill`: Skill categories and individual skills
- `Certification`: Title, issuer, dates
- `Project`: Title, description, technologies, URLs
- `Language`: Language and proficiency
- `Resume`: Complete resume object

### 2. `resume_parser.py`
Extracts text from resumes using PyMuPDF (and DOCX/TXT support):
- `ResumeParser.extract_text_from_pdf()`: Extract text from PDF
- `ResumeParser.extract_text_from_docx()`: Extract text from DOCX
- `ResumeParser.extract_text_from_txt()`: Extract text from TXT
- `ResumeParser.parse_resume()`: Main parsing method

### 3. `openrouter_client.py`
LLM integration via OpenRouter (DeepSeek v4 Flash):
- `OpenRouterClient.call_llm()`: Generic LLM API calls
- `OpenRouterClient.extract_resume_from_text()`: Parse resume text to JSON
- `OpenRouterClient.evaluate_candidate()`: Score candidate vs job description

### 4. `agentic_pipeline.py`
Multi-agent screening system:
- **PlannerAgent**: Decomposes screening query into search strategies
- **RetrievalAgent**: Semantic search using ChromaDB
- **CritiqueAgent**: Validates evaluations
- **ScreeningPipeline**: Orchestrates the complete flow

### 5. `main.py`
FastAPI REST server with endpoints:
- `POST /upload-resume`: Upload and parse single resume
- `POST /parse-resume-text`: Parse resume from raw text
- `POST /bulk-upload`: Upload multiple resumes
- `POST /screen-candidates`: Run full screening pipeline
- `GET /candidates`: List all candidates
- `GET /candidate/{id}`: Get candidate details
- `POST /evaluate-candidate/{id}`: Evaluate single candidate

## Installation

1. **Clone and setup**:
```bash
cd interview_agentic
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai)

## Usage

### Option 1: Command-line Examples

```bash
python example_usage.py
```

This runs three examples:
1. Parse a single resume
2. Screen multiple candidates (agentic pipeline)
3. Evaluate a single candidate

### Option 2: FastAPI Server

```bash
python main.py
```

Server starts at `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs`

**Example requests**:

```bash
# Upload a resume
curl -X POST "http://localhost:8000/upload-resume" \
  -F "file=@resume.pdf" \
  -F "candidate_id=john_doe"

# Parse resume from text
curl -X POST "http://localhost:8000/parse-resume-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "John Doe...", "candidate_id": "john"}'

# Run screening pipeline
curl -X POST "http://localhost:8000/screen-candidates" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python Developer...",
    "required_skills": ["Python", "Django", "React"],
    "min_experience_years": 5
  }'

# List all candidates
curl "http://localhost:8000/candidates"

# Get candidate details
curl "http://localhost:8000/candidate/john_doe"
```

### Option 3: Python Integration

```python
from resume_parser import ResumeParser
from openrouter_client import OpenRouterClient
from agentic_pipeline import ScreeningPipeline, ScreeningQuery

# Initialize
parser = ResumeParser()
llm = OpenRouterClient(api_key="your_key")
pipeline = ScreeningPipeline(api_key="your_key")

# Parse resume
resume = parser.parse_resume("resume.pdf", llm)

# Add to pipeline
pipeline.add_candidate(resume, "candidate_1")

# Screen
query = ScreeningQuery(
    job_description="Senior Developer needed",
    required_skills=["Python", "JavaScript"],
    min_experience_years=5
)
state = pipeline.screen_candidates(query)

# Get results
for ranking in state.final_rankings:
    print(f"Rank {ranking['rank']}: {ranking['match_score']}/100")
```

## Resume Fields Extracted

The system extracts and structures:

### Contact Information
- Name, email, phone
- Address, city, state, country
- LinkedIn, GitHub, portfolio, website

### Professional Profile
- Professional summary
- Skills (categorized)
- Languages and proficiency levels

### Experience
- Company, position, employment type
- Start/end dates, location
- Job description, achievements
- Technologies used

### Education
- Institution, degree, field of study
- Start/end dates, GPA
- Activities, additional description

### Additional
- Certifications (title, issuer, dates)
- Projects (title, description, technologies, URLs)
- Publications
- Volunteer experience
- Awards and recognition

## JSON Output Format

```json
{
  "contact_info": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-555-123-4567",
    "linkedin": "linkedin.com/in/johndoe",
    "github": "github.com/johndoe"
  },
  "professional_summary": "...",
  "skills": [
    {
      "category": "Programming Languages",
      "skills": ["Python", "JavaScript", "TypeScript"]
    }
  ],
  "experience": [
    {
      "company": "Tech Corp",
      "position": "Senior Engineer",
      "start_date": "2021-01",
      "end_date": "Present",
      "technologies": ["Python", "React", "AWS"],
      "achievements": ["Led team of 5", "Improved performance by 40%"]
    }
  ],
  "education": [
    {
      "institution": "IIT Bombay",
      "degree": "Bachelor of Technology",
      "field_of_study": "Computer Science",
      "gpa": "3.8/4.0"
    }
  ]
}
```

## Screening Pipeline Flow

1. **Intake**: Job description, required skills, experience level
2. **Planning**: LLM decomposes query into semantic search queries
3. **Retrieval**: ChromaDB finds relevant candidates
4. **Evaluation**: Each candidate scored against JD
5. **Critique**: Validations against requirements
6. **Ranking**: Final sorted list with recommendations

## Environment Variables

```
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=deepseek/deepseek-chat
CHROMA_DB_PATH=./chroma_db
```

## Performance Notes

- **ChromaDB**: Uses DuckDB backend with cosine similarity
- **Embeddings**: Generated by OpenRouter's embedding models
- **LLM**: DeepSeek v4 Flash for speed and cost-efficiency
- **Scalability**: Can handle 1000+ candidates efficiently

## Future Enhancements

- [ ] Interview question generation
- [ ] Automated email candidate outreach
- [ ] Interview scheduling
- [ ] Collaboration features (reviewer notes)
- [ ] Custom evaluation rubrics
- [ ] Bias detection in screening
- [ ] Resume quality scoring
- [ ] Competitive analysis (comparing candidates)

## Troubleshooting

**"OPENROUTER_API_KEY not set"**
- Add to .env file or export as environment variable

**"Failed to parse LLM response as JSON"**
- The LLM may have returned malformed JSON. Retry or adjust prompt.

**ChromaDB connection issues**
- Ensure `./chroma_db` directory is writable

**Resume parsing errors**
- Ensure file is valid PDF/DOCX/TXT
- Check file is not corrupted

## License

MIT

## Contact

For issues and questions, contact: chauhan.vkalidas.chy20@itbhu.ac.in
