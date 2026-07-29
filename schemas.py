from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ContactInfo:
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

    def dict(self):
        return asdict(self)


@dataclass
class Education:
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    activities: Optional[str] = None
    description: Optional[str] = None   

    def dict(self):
        return asdict(self)


@dataclass
class Experience:
    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    employment_type: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    achievements: Optional[List[str]] = field(default_factory=list)
    technologies: Optional[List[str]] = field(default_factory=list)

    def dict(self):
        return asdict(self)


@dataclass
class Skill:
    category: Optional[str] = None
    skills: List[str] = field(default_factory=list)

    def dict(self):
        return asdict(self)


@dataclass
class Certification:
    title: Optional[str] = None
    issuer: Optional[str] = None
    date_obtained: Optional[str] = None
    expiration_date: Optional[str] = None
    credential_url: Optional[str] = None

    def dict(self):
        return asdict(self)


@dataclass
class Project:
    title: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = field(default_factory=list)
    url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def dict(self):
        return asdict(self)


@dataclass
class Language:
    language: Optional[str] = None
    proficiency: Optional[str] = None

    def dict(self):
        return asdict(self)


@dataclass
class Resume:
    contact_info: Optional[ContactInfo] = field(default_factory=ContactInfo)
    professional_summary: Optional[str] = None
    skills: List[Skill] = field(default_factory=list)
    experience: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    certifications: List[Certification] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    languages: List[Language] = field(default_factory=list)
    publications: Optional[List[str]] = field(default_factory=list)
    volunteer_experience: Optional[List[str]] = field(default_factory=list)
    awards_recognition: Optional[List[str]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def dict(self):
        return {
            'contact_info': self.contact_info.dict() if self.contact_info else None,
            'professional_summary': self.professional_summary,
            'skills': [s.dict() for s in self.skills],
            'experience': [e.dict() for e in self.experience],
            'education': [ed.dict() for ed in self.education],
            'certifications': [c.dict() for c in self.certifications],
            'projects': [p.dict() for p in self.projects],
            'languages': [l.dict() for l in self.languages],
            'publications': self.publications,
            'volunteer_experience': self.volunteer_experience,
            'awards_recognition': self.awards_recognition,
            'metadata': self.metadata
        }
