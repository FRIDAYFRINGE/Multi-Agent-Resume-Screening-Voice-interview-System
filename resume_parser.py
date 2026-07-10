import fitz
import json
from typing import Optional
from schemas import Resume
import re


class ResumeParser:
    def __init__(self):
        self.extracted_text = ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            self.extracted_text = text
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX (basic implementation)"""
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            self.extracted_text = text
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.extracted_text = text
            return text
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    def parse_resume(self, file_path: str, llm_extractor=None) -> Resume:
        """
        Parse resume from file and extract structured data.
        If llm_extractor provided, uses LLM for extraction, otherwise uses regex.
        """
        # Determine file type and extract text
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = self.extract_text_from_docx(file_path)
        elif file_path.endswith('.txt'):
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT")

        if llm_extractor:
            return llm_extractor.extract_resume_from_text(text)
        else:
            return self._basic_extraction(text)

    def _basic_extraction(self, text: str) -> Resume:
        """Basic regex-based extraction (fallback)"""
        resume = Resume()

        # Email extraction
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match and resume.contact_info:
            resume.contact_info.email = email_match.group()

        # Phone extraction
        phone_match = re.search(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text)
        if phone_match and resume.contact_info:
            resume.contact_info.phone = phone_match.group()

        # Store raw text for LLM processing
        resume.metadata = {
            "raw_text": text[:1000],  # Store first 1000 chars as preview
            "extraction_method": "basic_regex"
        }

        return resume
