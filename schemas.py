from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    website: Optional[str] = None


class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    activities: Optional[str] = None
    description: Optional[str] = None


class Experience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    employment_type: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    achievements: Optional[List[str]] = None
    technologies: Optional[List[str]] = None


class Skill(BaseModel):
    category: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


class Certification(BaseModel):
    title: Optional[str] = None
    issuer: Optional[str] = None
    date_obtained: Optional[str] = None
    expiration_date: Optional[str] = None
    credential_url: Optional[str] = None


class Project(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = None
    url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class Language(BaseModel):
    language: Optional[str] = None
    proficiency: Optional[str] = None


class Resume(BaseModel):
    contact_info: Optional[ContactInfo] = Field(default_factory=ContactInfo)
    professional_summary: Optional[str] = None
    skills: List[Skill] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    publications: Optional[List[str]] = None
    volunteer_experience: Optional[List[str]] = None
    awards_recognition: Optional[List[str]] = None
    metadata: Optional[dict] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "contact_info": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "+1-123-456-7890"
                },
                "professional_summary": "...",
                "skills": [{"category": "Programming", "skills": ["Python", "JavaScript"]}],
                "experience": [{"company": "Tech Corp", "position": "Software Engineer"}]
            }
        }
