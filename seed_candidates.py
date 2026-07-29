"""
One-time script to seed ChromaDB with 50 synthetic candidates.
Run: python seed_candidates.py

Generates realistic resumes across 4 tiers relative to the default JD
(Senior AI/ML Engineer — Agentic Systems):
  - Tier 1 (15): Strong match  — LangGraph, RAG, agentic, LLM
  - Tier 2 (15): Good match    — solid ML, some relevant stack
  - Tier 3 (12): Weak match    — general software, ML adjacent
  - Tier 4 (8):  Poor match    — no ML, wrong domain
"""

import json, os, sys, time
from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from openrouter_client import OpenRouterClient
from agentic_pipeline import CandidateVectorStore, ScreeningPipeline
from schemas import (Resume, ContactInfo, Skill, Experience,
                     Education, Certification, Project)

# ── Candidate seeds ───────────────────────────────────────────────────────────
# (id, full_name, tier, summary, skills_by_category, experience_blurb,
#  education, years_exp)

SEEDS = [
    # ── Tier 1: Strong — agentic / RAG / LLM stack ──────────────────────────
    ("aditya_sharma", "Aditya Sharma", 1,
     "Senior ML Engineer specialising in agentic LLM systems and RAG pipelines. Built production multi-agent workflows with LangGraph and ChromaDB. 4 years experience.",
     {"AI/ML Frameworks": ["LangGraph", "LangChain", "CrewAI", "PyTorch", "Transformers"],
      "RAG & Vector DBs": ["ChromaDB", "FAISS", "Weaviate", "RAG", "Semantic Search"],
      "Backend": ["FastAPI", "Python", "Docker", "Redis"],
      "LLM": ["OpenAI API", "DeepSeek", "LoRA fine-tuning", "prompt engineering"]},
     [("Fintech AI Startup", "Senior ML Engineer", "2022-01", "Present",
       "Led design and deployment of a multi-agent LangGraph pipeline for document Q&A over 500K financial reports. Implemented hybrid BM25+dense retrieval with Reciprocal Rank Fusion, cutting hallucination rate by 60%. Built FastAPI microservices serving 50K requests/day.",
       ["LangGraph", "ChromaDB", "FastAPI", "PyTorch", "Docker"])],
     ("IIT Delhi", "B.Tech", "Computer Science", "2018", "2022", "8.7/10"),
     4),

    ("priya_nair", "Priya Nair", 1,
     "LLM fine-tuning specialist with 5 years in NLP research and production. Expert in LoRA, QLoRA, RLHF. Strong RAG pipeline and agentic system design background.",
     {"LLM": ["LoRA", "QLoRA", "RLHF", "DPO", "instruction tuning", "DeepSpeed"],
      "Frameworks": ["PyTorch", "Transformers", "LangGraph", "LangChain"],
      "Vector DB": ["FAISS", "ChromaDB", "Qdrant"],
      "MLOps": ["Weights & Biases", "MLflow", "Docker", "Kubernetes"]},
     [("AI Research Lab", "ML Research Engineer", "2021-06", "Present",
       "Fine-tuned Llama-2-7B and Mistral-7B for domain-specific Q&A using QLoRA on 4×A100 cluster. Implemented RLHF pipeline with human preference data, achieving 40% win-rate improvement vs base model. Built LangGraph agentic pipeline for automated literature review.",
       ["PyTorch", "LoRA", "QLoRA", "RLHF", "LangGraph", "FAISS"])],
     ("IISc Bangalore", "M.Tech", "AI & ML", "2019", "2021", "9.1/10"),
     5),

    ("rahul_verma", "Rahul Verma", 1,
     "Agentic systems architect with 6 years building production AI pipelines. Deep expertise in LangGraph state machines, CrewAI, tool-use agents, and multi-step reasoning systems.",
     {"Agentic Systems": ["LangGraph", "CrewAI", "AutoGPT patterns", "tool-use", "ReAct"],
      "RAG": ["ChromaDB", "Pinecone", "FAISS", "hybrid search", "reranking"],
      "Backend": ["FastAPI", "Python", "PostgreSQL", "Redis", "Celery"],
      "Cloud": ["AWS SageMaker", "Docker", "CI/CD", "Terraform"]},
     [("Enterprise AI Platform", "Principal AI Engineer", "2020-03", "Present",
       "Architected a 7-agent LangGraph pipeline for contract analysis: planner, retriever, extractor, validator, comparator, risk-scorer, and report-generator. Processes 10K contracts/month with 94% accuracy. Designed ChromaDB schema and embedding strategy for 2M+ document corpus.",
       ["LangGraph", "ChromaDB", "FastAPI", "PostgreSQL", "Docker", "AWS"])],
     ("NIT Trichy", "B.Tech", "Computer Science", "2015", "2019", "8.4/10"),
     6),

    ("alex_chen", "Alex Chen", 1,
     "ML platform engineer focused on LLM serving, RAG infrastructure, and developer tooling for AI teams. 5 years at high-scale ML companies.",
     {"ML Infra": ["vLLM", "TGI", "Triton Inference Server", "model quantization"],
      "RAG": ["LangGraph", "ChromaDB", "Weaviate", "embedding pipelines"],
      "Backend": ["FastAPI", "Python", "gRPC", "Kafka"],
      "DevOps": ["Kubernetes", "Docker", "Prometheus", "Grafana"]},
     [("ML Infrastructure Co.", "Senior ML Platform Engineer", "2021-01", "Present",
       "Built LLM serving platform handling 5M requests/day using vLLM with custom batching. Designed and deployed RAG platform using LangGraph orchestration and ChromaDB with 10M+ document index. Reduced p99 latency by 45% through model quantization and batching optimisation.",
       ["vLLM", "LangGraph", "ChromaDB", "FastAPI", "Kubernetes", "Python"])],
     ("Stanford University", "M.S.", "Computer Science", "2017", "2019", "3.9/4.0"),
     5),

    ("sarah_kim", "Sarah Kim", 1,
     "NLP researcher turned production ML engineer. 4 years publishing and deploying transformer-based systems. Expert in attention mechanisms, RAG, and evaluation frameworks.",
     {"NLP/LLM": ["Transformers", "BERT", "GPT", "RAG", "semantic search", "NER"],
      "Frameworks": ["PyTorch", "HuggingFace", "LangChain", "LangGraph"],
      "Vector DB": ["FAISS", "ChromaDB", "Elasticsearch"],
      "Backend": ["FastAPI", "Python", "PostgreSQL"]},
     [("NLP Research Group", "ML Engineer / Researcher", "2020-09", "Present",
       "Published 3 papers on retrieval-augmented generation at ACL and EMNLP. Deployed production RAG system for enterprise knowledge base with LangGraph orchestration, serving 20K daily queries with 91% relevance score. Open-sourced evaluation framework used by 500+ teams.",
       ["PyTorch", "Transformers", "RAG", "LangGraph", "ChromaDB", "FastAPI"])],
     ("Seoul National University", "Ph.D.", "Computational Linguistics", "2016", "2020", "4.0/4.0"),
     4),

    ("miguel_rodriguez", "Miguel Rodriguez", 1,
     "RAG pipeline specialist with 4 years building enterprise search and Q&A systems. Expert in chunking strategies, embedding models, hybrid retrieval, and reranking.",
     {"RAG Stack": ["LangGraph", "LangChain", "ChromaDB", "FAISS", "Pinecone", "BM25"],
      "Retrieval": ["hybrid search", "reranking", "RRF", "cross-encoders", "DPR"],
      "Backend": ["FastAPI", "Python", "Elasticsearch", "PostgreSQL"],
      "LLM": ["OpenAI", "Anthropic Claude", "prompt engineering", "evaluation"]},
     [("Enterprise Search Co.", "Senior AI Engineer", "2021-06", "Present",
       "Built RAG platform for Fortune 500 legal firm over 5M documents. Implemented semantic chunking with 15% overlap, hybrid BM25+dense retrieval, and cross-encoder reranking — improved answer accuracy from 67% to 88%. Deployed as LangGraph pipeline with FastAPI backend.",
       ["LangGraph", "ChromaDB", "BM25", "reranking", "FastAPI", "Python"])],
     ("Universidad Complutense", "M.S.", "Data Science", "2018", "2020", "8.9/10"),
     4),

    ("arjun_patel", "Arjun Patel", 1,
     "Full-stack ML engineer with 5 years. Builds end-to-end AI products: model training, RAG pipelines, FastAPI backends, and React frontends. Active open-source contributor.",
     {"ML": ["PyTorch", "fine-tuning", "LoRA", "RAG", "LangGraph", "ChromaDB"],
      "Backend": ["FastAPI", "Python", "Django", "PostgreSQL", "Redis"],
      "Frontend": ["React", "TypeScript", "Next.js"],
      "DevOps": ["Docker", "AWS", "GitHub Actions"]},
     [("AI Product Studio", "ML Engineer", "2020-01", "Present",
       "Built SaaS AI writing assistant: fine-tuned GPT model with LoRA, RAG knowledge base using ChromaDB, FastAPI backend, React frontend. 10K paying users. Also built multi-agent document processor using LangGraph for automated data extraction and validation.",
       ["PyTorch", "LoRA", "ChromaDB", "LangGraph", "FastAPI", "React"])],
     ("BITS Pilani", "B.E.", "Computer Science", "2016", "2020", "8.2/10"),
     5),

    ("liu_yang", "Liu Yang", 1,
     "Vector database and embedding systems specialist, 4 years. Expert in ChromaDB, Weaviate, Pinecone internals, HNSW indexing, and production embedding pipelines.",
     {"Vector DB": ["ChromaDB", "Weaviate", "Pinecone", "Milvus", "HNSW", "IVF"],
      "Embeddings": ["sentence-transformers", "OpenAI ada-002", "E5", "BGE", "fine-tuned embeddings"],
      "ML": ["LangGraph", "LangChain", "PyTorch", "semantic search"],
      "Backend": ["FastAPI", "Python", "Go", "Redis"]},
     [("Search Infrastructure Co.", "Senior Engineer", "2021-03", "Present",
       "Designed and scaled ChromaDB deployment to 50M vectors with custom HNSW parameters. Built embedding fine-tuning pipeline for domain adaptation, improving retrieval recall@10 from 71% to 89%. Integrated LangGraph orchestration layer for multi-step RAG workflows.",
       ["ChromaDB", "HNSW", "sentence-transformers", "LangGraph", "Python", "Go"])],
     ("Tsinghua University", "M.S.", "Computer Science", "2019", "2021", "88/100"),
     4),

    ("emily_johnson", "Emily Johnson", 1,
     "AI infrastructure engineer with 5 years specialising in LLMOps, model serving, and production agentic systems. Reduced ML deployment cycle from 2 weeks to 4 hours.",
     {"LLMOps": ["MLflow", "Weights & Biases", "LangSmith", "model monitoring"],
      "Agentic": ["LangGraph", "CrewAI", "tool-use", "function calling"],
      "Infra": ["Kubernetes", "Docker", "AWS SageMaker", "Ray Serve"],
      "Backend": ["FastAPI", "Python", "Airflow"]},
     [("AI Platform Team", "Senior MLOps Engineer", "2020-06", "Present",
       "Built LLMOps platform for 15-person ML team: automated evaluation pipelines, LangSmith tracing integration, A/B testing for prompt versions, and deployment automation. Deployed production LangGraph multi-agent systems with full observability. Zero-downtime model updates.",
       ["LangGraph", "LangSmith", "FastAPI", "Kubernetes", "MLflow", "Python"])],
     ("Georgia Tech", "M.S.", "Computer Science", "2017", "2019", "3.8/4.0"),
     5),

    ("hassan_ali", "Hassan Ali", 1,
     "Multimodal AI engineer with 3 years. Combines vision and language models in production RAG systems. Expert in document AI, OCR, and structured data extraction from unstructured sources.",
     {"Multimodal": ["GPT-4V", "LLaVA", "document AI", "OCR", "table extraction"],
      "RAG": ["LangGraph", "ChromaDB", "unstructured.io", "document parsing"],
      "Backend": ["FastAPI", "Python", "PostgreSQL"],
      "ML": ["PyTorch", "Transformers", "fine-tuning"]},
     [("Document AI Startup", "AI Engineer", "2022-01", "Present",
       "Built multimodal RAG system processing PDFs, images, and tables using LLaVA for visual understanding and LangGraph for orchestration. ChromaDB stores text+image embeddings. Processes 100K documents/day for enterprise clients. Improved extraction accuracy by 35% vs pure-text baseline.",
       ["LangGraph", "ChromaDB", "LLaVA", "FastAPI", "PyTorch", "unstructured.io"])],
     ("Cairo University", "B.Sc.", "Computer Engineering", "2018", "2022", "3.7/4.0"),
     3),

    ("nina_petrov", "Nina Petrov", 1,
     "ML engineer specialising in production LLM systems, prompt engineering, and evaluation. 4 years at AI-first companies. Built automated eval frameworks and RLHF pipelines.",
     {"LLM": ["prompt engineering", "chain-of-thought", "RLHF", "DPO", "evaluation"],
      "Frameworks": ["LangGraph", "LangChain", "OpenAI", "Anthropic"],
      "Vector DB": ["ChromaDB", "Qdrant", "pgvector"],
      "Backend": ["FastAPI", "Python", "PostgreSQL"]},
     [("LLM Products Co.", "ML Engineer", "2021-04", "Present",
       "Built automated LLM evaluation suite with 500+ test cases, human preference alignment scores, and regression tracking. Designed multi-agent LangGraph pipeline for customer support: intent classifier, retriever, answer generator, fact-checker. 92% customer satisfaction improvement.",
       ["LangGraph", "ChromaDB", "RLHF", "FastAPI", "Python", "PostgreSQL"])],
     ("Moscow State University", "M.S.", "Applied Mathematics", "2017", "2019", "5.0/5.0"),
     4),

    ("carlos_mendez", "Carlos Mendez", 1,
     "LLM researcher and engineer with 4 years in fine-tuning, alignment, and production deployment. Focused on efficient training: QLoRA, PEFT, knowledge distillation.",
     {"Fine-tuning": ["LoRA", "QLoRA", "PEFT", "knowledge distillation", "RLHF", "DPO"],
      "Frameworks": ["PyTorch", "DeepSpeed", "Transformers", "Accelerate"],
      "RAG": ["LangChain", "LangGraph", "FAISS", "ChromaDB"],
      "MLOps": ["MLflow", "Docker", "AWS", "Weights & Biases"]},
     [("AI Research Institute", "Research Engineer", "2021-01", "Present",
       "Fine-tuned Llama-2-13B using QLoRA achieving GPT-3.5-level performance at 10% cost. Built RLHF data collection pipeline with 50K preference annotations. Deployed model serving system with FastAPI and LangGraph orchestration for production agentic workflows.",
       ["PyTorch", "QLoRA", "RLHF", "LangGraph", "FastAPI", "DeepSpeed"])],
     ("Universidad de Buenos Aires", "M.S.", "Computer Science", "2018", "2020", "9.2/10"),
     4),

    ("yuki_tanaka", "Yuki Tanaka", 1,
     "Agentic AI engineer with 4 years. Built production multi-agent systems using LangGraph and LangChain. Expert in tool-use, function calling, memory systems, and agent evaluation.",
     {"Agentic": ["LangGraph", "LangChain", "AutoGen", "tool-use", "function calling", "memory"],
      "LLM": ["GPT-4", "Claude", "prompt chaining", "structured outputs"],
      "Vector DB": ["ChromaDB", "FAISS", "Qdrant"],
      "Backend": ["FastAPI", "Python", "PostgreSQL", "Docker"]},
     [("Automation AI Startup", "Senior AI Engineer", "2021-07", "Present",
       "Built 12-agent LangGraph workflow for software development automation: planner, coder, reviewer, tester, documenter agents. System generates, tests, and deploys code changes autonomously. Reduced developer workload by 40%. Integrated ChromaDB memory for context persistence.",
       ["LangGraph", "ChromaDB", "FastAPI", "Python", "Docker"])],
     ("Keio University", "B.Eng.", "Information Engineering", "2017", "2021", "3.8/4.0"),
     4),

    ("fatima_malik", "Fatima Malik", 1,
     "AI engineer with 3 years combining computer vision and LLMs in production. Built vision-language pipelines for medical imaging and industrial inspection with RAG support.",
     {"Multimodal": ["CLIP", "ViT", "GPT-4V", "vision-language models"],
      "RAG/Agents": ["LangGraph", "LangChain", "ChromaDB", "RAG"],
      "CV": ["PyTorch", "OpenCV", "YOLO", "segmentation"],
      "Backend": ["FastAPI", "Python", "Docker"]},
     [("HealthTech AI", "ML Engineer", "2022-06", "Present",
       "Built vision-language RAG pipeline for medical report generation: CLIP image embeddings + text in ChromaDB, LangGraph orchestration, GPT-4V for reasoning. Reduced radiologist report time by 30%. FastAPI backend processing 5K scans/day.",
       ["LangGraph", "ChromaDB", "CLIP", "GPT-4V", "FastAPI", "PyTorch"])],
     ("NUST Islamabad", "B.E.", "Computer Systems Engineering", "2018", "2022", "3.8/4.0"),
     3),

    ("james_okafor", "James Okafor", 1,
     "ML research engineer with 4 years bridging academic NLP research and production systems. Expert in information retrieval, knowledge graphs, and conversational AI.",
     {"NLP": ["information retrieval", "knowledge graphs", "NER", "RE", "coreference"],
      "RAG": ["LangGraph", "ChromaDB", "hybrid search", "dense retrieval"],
      "ML": ["PyTorch", "Transformers", "spaCy", "NLTK"],
      "Backend": ["FastAPI", "Python", "Neo4j", "PostgreSQL"]},
     [("AI Research Lab", "Research Engineer", "2020-09", "Present",
       "Built knowledge-graph-augmented RAG system combining ChromaDB vector search with Neo4j entity relationships. LangGraph orchestration handles multi-hop reasoning over 2M entities. Published at NAACL. FastAPI backend serving researchers at 5 universities.",
       ["LangGraph", "ChromaDB", "Neo4j", "FastAPI", "PyTorch", "Transformers"])],
     ("University of Lagos", "M.S.", "Computer Science", "2018", "2020", "4.0/5.0"),
     4),

    # ── Tier 2: Good — solid ML but partial stack match ───────────────────────
    ("michael_brown", "Michael Brown", 2,
     "Senior backend Python engineer with 3 years of ML integration experience. Strong FastAPI and PostgreSQL background. Currently building ML-powered features without deep model expertise.",
     {"Backend": ["FastAPI", "Python", "PostgreSQL", "Redis", "Celery"],
      "ML Adjacent": ["scikit-learn", "pandas", "OpenAI API integration", "basic RAG"],
      "DevOps": ["Docker", "AWS", "GitHub Actions"]},
     [("SaaS Platform", "Senior Backend Engineer", "2020-06", "Present",
       "Built FastAPI microservices powering 200K daily users. Integrated OpenAI GPT-4 for content generation features using basic prompt chaining. Implemented simple RAG with pgvector for knowledge-base search. No experience with LangGraph or advanced agentic systems.",
       ["FastAPI", "Python", "PostgreSQL", "OpenAI API", "pgvector", "Docker"])],
     ("University of Manchester", "B.Sc.", "Software Engineering", "2015", "2019", "2.1/First"),
     5),

    ("jennifer_lee", "Jennifer Lee", 2,
     "Data scientist with 4 years transitioning to ML engineering. Strong statistical ML background. Building first production LLM applications, learning LangGraph and RAG architecture.",
     {"Data Science": ["scikit-learn", "pandas", "NumPy", "statistical modelling", "A/B testing"],
      "ML": ["PyTorch basics", "XGBoost", "feature engineering"],
      "LLM (learning)": ["LangChain basics", "OpenAI API", "basic prompt engineering"],
      "Tools": ["Python", "SQL", "Jupyter", "Git"]},
     [("Analytics Co.", "Senior Data Scientist", "2020-01", "Present",
       "Built churn prediction models saving $2M annually. Recently led pilot LLM project using LangChain for customer FAQ automation — basic RAG with pgvector. No production experience with LangGraph, ChromaDB, or agentic pipelines yet.",
       ["Python", "scikit-learn", "LangChain", "OpenAI API", "PostgreSQL"])],
     ("UC Berkeley", "M.S.", "Statistics", "2017", "2019", "3.6/4.0"),
     4),

    ("david_wilson", "David Wilson", 2,
     "Python backend engineer with growing ML interest. 5 years building APIs and data pipelines. Started building LLM applications in the past year using LangChain and basic RAG.",
     {"Backend": ["Python", "FastAPI", "Flask", "PostgreSQL", "MongoDB", "Redis"],
      "LLM (beginner)": ["LangChain", "OpenAI API", "basic embeddings", "vector search"],
      "Data": ["pandas", "ETL pipelines", "Apache Kafka"]},
     [("Data Platform Co.", "Backend Engineer", "2019-03", "Present",
       "Built high-throughput data ingestion pipelines processing 10M events/day. In last 6 months built internal LangChain chatbot over company docs using basic FAISS-backed RAG. First LLM project — learning agentic patterns but no LangGraph production experience.",
       ["Python", "FastAPI", "LangChain", "FAISS", "PostgreSQL", "Kafka"])],
     ("University of Leeds", "B.Sc.", "Computer Science", "2014", "2018", "2:1"),
     5),

    ("aisha_ibrahim", "Aisha Ibrahim", 2,
     "MLOps engineer with 3 years. Expert in model training infrastructure, CI/CD for ML, and monitoring. Less experience with LLMs and agentic systems specifically.",
     {"MLOps": ["MLflow", "Kubeflow", "DVC", "model monitoring", "feature stores"],
      "ML": ["PyTorch", "scikit-learn", "XGBoost", "training pipelines"],
      "Infra": ["Kubernetes", "Docker", "Terraform", "AWS SageMaker"],
      "LLM (some)": ["Weights & Biases", "basic LangChain", "HuggingFace Hub"]},
     [("ML Platform Team", "MLOps Engineer", "2021-09", "Present",
       "Built ML training infrastructure for 10-person team: automated retraining, model registry, A/B deployment, drift detection. Recently involved in deploying first LLM — used LangChain basic chain, no LangGraph or agentic patterns. Strong on infra, lighter on LLM-specific knowledge.",
       ["Kubeflow", "MLflow", "PyTorch", "Kubernetes", "Docker", "AWS"])],
     ("University of Nairobi", "B.Sc.", "Computer Science", "2018", "2022", "First Class"),
     3),

    ("peter_kozlov", "Peter Kozlov", 2,
     "Data engineer with 5 years managing large-scale data infrastructure. Strong Python, Spark, and ETL. Recently upskilling into ML engineering with focus on data pipelines for LLM training.",
     {"Data Engineering": ["Apache Spark", "Airflow", "dbt", "Kafka", "Snowflake", "Delta Lake"],
      "Python": ["Python", "pandas", "SQLAlchemy", "FastAPI basics"],
      "Cloud": ["AWS Glue", "GCP BigQuery", "Databricks"],
      "ML (learning)": ["basic embeddings", "data pipelines for LLM fine-tuning"]},
     [("Data Platform Team", "Senior Data Engineer", "2019-07", "Present",
       "Built petabyte-scale data lake on AWS. Designed data pipelines feeding ML training jobs. Recently built data preparation pipeline for LLM fine-tuning dataset (tokenization, dedup, quality filtering). No experience with LangGraph, RAG serving, or agentic systems.",
       ["Apache Spark", "Airflow", "Python", "AWS", "Snowflake", "dbt"])],
     ("Moscow Institute of Physics", "M.S.", "Applied Math", "2014", "2016", "4.8/5.0"),
     5),

    ("sophie_martin", "Sophie Martin", 2,
     "Senior Python developer with 6 years. Strong software engineering fundamentals. Built 2 ML-powered features at current company. Interested in moving into dedicated ML engineering role.",
     {"Backend": ["Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Celery"],
      "ML Applied": ["scikit-learn", "pandas", "OpenAI API", "basic vector search"],
      "DevOps": ["Docker", "CI/CD", "AWS EC2", "GitHub Actions"]},
     [("B2B SaaS Co.", "Senior Backend Engineer", "2018-04", "Present",
       "Senior engineer on core platform team. Led integration of GPT-4 for document summarisation (direct API, no RAG). Built product recommendation engine using collaborative filtering. No deep ML research background but strong engineering skills and eager to specialise.",
       ["Python", "FastAPI", "PostgreSQL", "OpenAI API", "Docker", "Redis"])],
     ("Ecole Polytechnique", "M.Eng.", "Computer Science", "2014", "2018", "16/20"),
     6),

    ("raj_krishnan", "Raj Krishnan", 2,
     "Data scientist turned ML engineer with 4 years. Built statistical models and basic ML pipelines. Now focused on LLM applications and exploring LangGraph for production workflows.",
     {"Data Science": ["scikit-learn", "statsmodels", "pandas", "time series", "clustering"],
      "ML Engineering": ["PyTorch", "FastAPI", "model serving", "basic MLOps"],
      "LLM": ["LangChain", "OpenAI API", "embeddings", "basic RAG with FAISS"]},
     [("Fintech Co.", "ML Engineer", "2020-09", "Present",
       "Built credit scoring model reducing default rate by 18%. Deployed FastAPI model serving. In last year started building LLM features: LangChain-based contract clause extractor using FAISS RAG. No LangGraph experience; has read the docs and wants to transition.",
       ["Python", "PyTorch", "FastAPI", "LangChain", "FAISS", "scikit-learn"])],
     ("IIT Bombay", "M.Tech", "Data Science", "2018", "2020", "8.1/10"),
     4),

    ("anna_novak", "Anna Novak", 2,
     "ML researcher with 5 years in academia transitioning to industry. Strong theoretical background in NLP and deep learning. Limited production engineering experience.",
     {"Research": ["NLP", "transformers", "attention mechanisms", "knowledge graphs"],
      "Frameworks": ["PyTorch", "HuggingFace", "spaCy", "NLTK"],
      "LLM": ["fine-tuning", "RLHF theory", "RAG concepts", "LangChain intro"],
      "Tools": ["Python", "Git", "LaTeX", "Jupyter"]},
     [("University Research Lab", "PhD Researcher / Postdoc", "2018-09", "Present",
       "5 publications at ACL, EMNLP, NAACL on cross-lingual transfer and knowledge-grounded generation. Strong theoretical understanding of RAG and agentic systems. Limited production FastAPI/deployment experience — transitioning from research to industry.",
       ["PyTorch", "HuggingFace", "spaCy", "Python", "research workflows"])],
     ("Charles University Prague", "Ph.D.", "Computational Linguistics", "2015", "2020", "Distinction"),
     5),

    ("tom_nguyen", "Tom Nguyen", 2,
     "Full-stack engineer with 4 years and strong ML side projects. Built personal RAG apps and fine-tuned models. Looking to transition into a dedicated ML engineering role.",
     {"Full-Stack": ["React", "Node.js", "Python", "FastAPI", "PostgreSQL"],
      "ML Projects": ["PyTorch", "HuggingFace", "LangChain", "FAISS", "basic fine-tuning"],
      "Cloud": ["AWS", "Vercel", "Docker"]},
     [("Product Startup", "Full-Stack Engineer", "2020-06", "Present",
       "Built React/FastAPI SaaS product. As side projects: fine-tuned GPT-2 for creative writing, built personal RAG chatbot over documentation using LangChain + FAISS. No production LLM or agentic experience at work. ML skills are self-taught and project-based.",
       ["React", "FastAPI", "Python", "LangChain", "FAISS", "PyTorch"])],
     ("University of Toronto", "B.Sc.", "Computer Science", "2016", "2020", "3.5/4.0"),
     4),

    ("maria_garcia", "Maria Garcia", 2,
     "NLP engineer with 4 years building text processing pipelines. Strong in classical NLP and early transformer models. Currently upskilling on LLMs and agentic frameworks.",
     {"NLP": ["spaCy", "NLTK", "BERT", "text classification", "NER", "summarisation"],
      "ML": ["PyTorch", "scikit-learn", "HuggingFace"],
      "Backend": ["FastAPI", "Python", "Elasticsearch"],
      "LLM (learning)": ["GPT API", "basic prompting", "LangChain intro"]},
     [("Search & NLP Co.", "NLP Engineer", "2020-03", "Present",
       "Built text classification and NER pipelines for e-commerce product catalogue (5M items). Implemented semantic search with BERT embeddings and Elasticsearch. Recently integrated GPT-4 via API for product descriptions. No experience with LangGraph, ChromaDB, or agentic systems.",
       ["spaCy", "BERT", "PyTorch", "FastAPI", "Elasticsearch", "Python"])],
     ("Universidad Politecnica Madrid", "M.S.", "AI", "2018", "2020", "8.7/10"),
     4),

    ("kevin_white", "Kevin White", 2,
     "Cloud ML engineer with 4 years deploying and scaling ML models on AWS. Strong deployment and monitoring skills. Less experience in model development and agentic systems.",
     {"Cloud ML": ["AWS SageMaker", "Lambda", "ECS", "Step Functions", "S3"],
      "ML Ops": ["model deployment", "A/B testing", "monitoring", "feature stores"],
      "Backend": ["Python", "FastAPI", "Docker", "Terraform"],
      "LLM": ["Bedrock", "basic RAG", "prompt engineering basics"]},
     [("AWS ML Platform", "Cloud ML Engineer", "2020-11", "Present",
       "Deployed and monitored 20+ ML models on SageMaker serving 1M+ daily predictions. Recently deployed RAG chatbot using AWS Bedrock and basic vector search. No LangGraph or ChromaDB experience. Strong on deployment and scaling, weaker on model development.",
       ["AWS SageMaker", "Python", "FastAPI", "Docker", "Bedrock", "Terraform"])],
     ("University of Washington", "B.S.", "Computer Science", "2016", "2020", "3.4/4.0"),
     4),

    # ── Tier 3: Weak match — general software, no ML focus ────────────────────
    ("chris_taylor", "Chris Taylor", 3,
     "Senior Java/Python backend engineer with 7 years. Expert in microservices and distributed systems. No ML or AI experience. Interested in moving into AI-adjacent roles.",
     {"Backend": ["Java", "Spring Boot", "Python", "FastAPI basics", "PostgreSQL", "Kafka"],
      "Architecture": ["microservices", "event-driven", "REST APIs", "gRPC"],
      "DevOps": ["Docker", "Kubernetes", "AWS", "CI/CD"]},
     [("Enterprise Software Co.", "Senior Backend Engineer", "2017-01", "Present",
       "Led design of microservices platform processing 5M transactions/day. Java/Spring Boot primary stack. Some Python scripting. No ML, LLM, RAG, or AI systems experience. Applying to pivot into AI engineering.",
       ["Java", "Spring Boot", "Python", "PostgreSQL", "Kafka", "Kubernetes"])],
     ("University of Birmingham", "B.Sc.", "Computer Science", "2013", "2017", "2:1"),
     7),

    ("jessica_moore", "Jessica Moore", 3,
     "Frontend engineer with 5 years building React applications. Recently added Python FastAPI to skill set. No ML experience. Interested in AI product development.",
     {"Frontend": ["React", "TypeScript", "Next.js", "Vue.js", "CSS/Tailwind"],
      "Backend (some)": ["Python", "FastAPI basics", "REST APIs", "PostgreSQL basics"],
      "Tools": ["Git", "Figma", "Jest", "Webpack"]},
     [("Product Agency", "Senior Frontend Engineer", "2019-06", "Present",
       "Led frontend development for 8 SaaS products. Recently built simple AI chatbot UI connecting to OpenAI API (direct calls, no RAG). No ML engineering background. Applying to transition to AI product engineering.",
       ["React", "TypeScript", "Python", "FastAPI", "OpenAI API"])],
     ("Durham University", "B.Sc.", "Digital Media", "2014", "2018", "2:1"),
     5),

    ("ryan_anderson", "Ryan Anderson", 3,
     "DevOps/SRE engineer with 5 years. Expert in Kubernetes, Terraform, and CI/CD. No ML background. Looking to move into ML infrastructure roles.",
     {"DevOps": ["Kubernetes", "Docker", "Terraform", "Ansible", "Prometheus", "Grafana"],
      "Cloud": ["AWS", "GCP", "Azure", "Helm", "ArgoCD"],
      "Scripting": ["Python", "Bash", "Go basics"]},
     [("Platform Engineering", "Senior DevOps Engineer", "2019-04", "Present",
       "Manages Kubernetes clusters with 500+ microservices. Built CI/CD platform reducing deployment time 80%. No ML or AI project experience. Applying to pivot to ML infrastructure.",
       ["Kubernetes", "Terraform", "Python", "AWS", "Prometheus", "Docker"])],
     ("Heriot-Watt University", "B.Eng.", "Systems Engineering", "2014", "2019", "2:1"),
     5),

    ("marcus_jackson", "Marcus Jackson", 3,
     "Data analyst with 4 years working in BI and reporting. Strong SQL and Excel. Some pandas experience. No ML engineering or LLM experience.",
     {"Analytics": ["SQL", "Excel", "Tableau", "Power BI", "Google Analytics"],
      "Python (some)": ["pandas", "matplotlib", "Jupyter notebooks", "basic scripts"]},
     [("Retail Analytics Co.", "Senior Data Analyst", "2020-01", "Present",
       "Built dashboards and reports for merchandising team. Wrote SQL queries over large datasets. Some pandas scripting for data cleaning. No ML models built. Applied for ML role to upskill.",
       ["SQL", "Python", "pandas", "Tableau", "Excel"])],
     ("University of Nottingham", "B.Sc.", "Business Analytics", "2016", "2020", "2:1"),
     4),

    ("claire_thompson", "Claire Thompson", 3,
     "Data engineer with 5 years in Spark and ETL pipelines. Excellent data infrastructure skills. No ML model development or LLM experience.",
     {"Data Engineering": ["Apache Spark", "Databricks", "Airflow", "dbt", "Kafka"],
      "Python": ["Python", "PySpark", "pandas", "FastAPI basics"],
      "Cloud": ["AWS", "GCP", "Snowflake"]},
     [("Data Platform", "Data Engineer", "2019-08", "Present",
       "Built Spark pipelines processing 50TB/day. Expert in data quality, lineage, and orchestration. Recently added FastAPI endpoint for data access. No ML, LLM, or AI experience. Wants to move into ML data infrastructure.",
       ["Apache Spark", "Airflow", "Python", "Databricks", "Snowflake"])],
     ("University of Edinburgh", "M.Sc.", "Data Science (coursework)", "2017", "2018", "Merit"),
     5),

    ("alex_williams", "Alex Williams", 3,
     "Cloud solutions architect with 6 years at AWS and client projects. No ML experience. Advising clients on AI product strategy but never built ML systems.",
     {"Cloud": ["AWS", "Azure", "GCP", "Terraform", "CDK", "serverless"],
      "Architecture": ["system design", "cost optimisation", "security"],
      "Programming": ["Python scripting", "JavaScript", "Bash"]},
     [("Cloud Consultancy", "Solutions Architect", "2018-01", "Present",
       "Led architecture for 40+ enterprise cloud migrations. Advising 3 clients on AI/ML adoption strategy. No hands-on ML development. AWS ML Specialty certification in progress.",
       ["AWS", "Terraform", "Python", "CDK", "Azure"])],
     ("University of Bristol", "B.Sc.", "Computer Science", "2012", "2016", "2:1"),
     6),

    ("daniel_harris", "Daniel Harris", 3,
     "Backend Go and Python developer with 4 years. Built high-performance APIs. No ML experience but interested in AI engineering.",
     {"Backend": ["Go", "Python", "gRPC", "REST", "PostgreSQL", "Redis", "Kafka"],
      "DevOps": ["Docker", "Kubernetes", "AWS"]},
     [("API Platform", "Backend Engineer", "2020-09", "Present",
       "Built Go microservices handling 100K RPS. Strong systems programming background. No ML or AI project experience. Applied to pivot into ML engineering given interest in the field.",
       ["Go", "Python", "PostgreSQL", "Kafka", "Kubernetes"])],
     ("Trinity College Dublin", "B.Sc.", "Computer Science", "2016", "2020", "First Class"),
     4),

    ("maya_jones", "Maya Jones", 3,
     "Software architect with 8 years. Expert in system design and engineering leadership. Limited ML knowledge — managed an ML team but didn't build models.",
     {"Architecture": ["system design", "DDD", "event sourcing", "CQRS"],
      "Tech": ["Python", "Java", "PostgreSQL", "Kafka", "Docker"],
      "Leadership": ["tech lead", "architecture reviews", "team mentoring"]},
     [("Enterprise Co.", "Principal Software Architect", "2016-01", "Present",
       "Led architecture for 60-person engineering org. Managed data science team of 5 (non-technical management). Strong system design skills but no hands-on ML or LLM experience. Interested in agentic system architecture roles.",
       ["Python", "Java", "PostgreSQL", "Kafka", "Docker"])],
     ("Imperial College London", "M.Eng.", "Computing", "2011", "2015", "Merit"),
     8),

    ("ben_patel", "Ben Patel", 3,
     "Site reliability and platform engineer with 5 years. Strong Kubernetes and observability skills. No ML background. Interested in AI infrastructure roles.",
     {"SRE": ["Kubernetes", "Prometheus", "Grafana", "Jaeger", "PagerDuty"],
      "Platform": ["Terraform", "Helm", "ArgoCD", "GitHub Actions"],
      "Programming": ["Python", "Go", "Bash"],
      "Cloud": ["GCP", "AWS"]},
     [("Platform Engineering", "Senior SRE", "2019-06", "Present",
       "Manages reliability for 200+ services. Reduced MTTR from 45min to 8min via improved observability. No ML experience. Applying to move into ML platform engineering.",
       ["Kubernetes", "Python", "Go", "Terraform", "GCP"])],
     ("University of Warwick", "B.Sc.", "Computer Science", "2014", "2019", "First"),
     5),

    ("sara_white", "Sara White", 3,
     "Python developer with 3 years. Built web applications and data scripts. No ML or AI experience. Recently completed Andrew Ng's ML course online.",
     {"Backend": ["Python", "Django", "Flask", "PostgreSQL", "REST APIs"],
      "Learning": ["machine learning basics", "linear regression", "neural networks (theory)"]},
     [("Web Agency", "Python Developer", "2021-06", "Present",
       "Built web apps and REST APIs for small businesses. No production ML experience. Currently studying ML through online courses. Applying for junior ML engineering role.",
       ["Python", "Django", "PostgreSQL", "REST"])],
     ("Leeds Beckett University", "B.Sc.", "Software Development", "2018", "2021", "2:1"),
     3),

    ("john_davis", "John Davis", 3,
     "Senior Java enterprise developer with 9 years at financial services companies. No ML experience. Interested in modernising to Python and AI stacks.",
     {"Java": ["Java", "Spring Boot", "Hibernate", "Maven", "JUnit"],
      "Enterprise": ["Oracle DB", "IBM MQ", "COBOL interfaces", "SOA"],
      "Some Python": ["Python scripting", "basic Flask"]},
     [("Investment Bank", "Senior Java Developer", "2015-03", "Present",
       "Maintains trading system processing £500M/day. 9 years Java enterprise experience. No ML, Python production, or AI background. Applying to reskill to AI engineering — strong fundamentals.",
       ["Java", "Spring Boot", "Oracle", "Python basics"])],
     ("University of Glasgow", "B.Sc.", "Software Engineering", "2010", "2014", "2:1"),
     9),

    # ── Tier 4: Poor match — wrong domain ─────────────────────────────────────
    ("steve_miller", "Steve Miller", 4,
     "Senior React and TypeScript frontend engineer with 7 years. No backend, no ML, no Python experience.",
     {"Frontend": ["React", "TypeScript", "Next.js", "Redux", "CSS/SCSS", "Webpack"]},
     [("Digital Agency", "Senior Frontend Engineer", "2017-01", "Present",
       "Leads frontend development for 15+ web applications. Purely frontend — no backend, database, or ML experience.",
       ["React", "TypeScript", "Next.js"])],
     ("Staffordshire University", "B.A.", "Graphic Design / Web", "2013", "2017", "2:1"),
     7),

    ("robert_taylor", "Robert Taylor", 4,
     "Java enterprise architect with 15 years. Legacy systems expert. No ML, Python, or modern AI stack knowledge.",
     {"Enterprise Java": ["Java EE", "EJB", "JBoss", "Oracle DB", "SOAP/WSDL", "XML"],
      "Architecture": ["SOA", "ESB", "legacy modernisation"]},
     [("Legacy Systems Consultancy", "Enterprise Architect", "2009-01", "Present",
       "15 years maintaining and modernising Java EE systems for large enterprises. No Python, ML, or AI experience whatsoever.",
       ["Java EE", "Oracle", "JBoss", "SOAP"])],
     ("University of Sheffield", "B.Sc.", "Computer Science", "2005", "2009", "2:2"),
     15),

    ("karen_brown", "Karen Brown", 4,
     "Database administrator with 10 years. Expert in Oracle, PostgreSQL, and SQL Server administration. No development or ML experience.",
     {"DBA": ["Oracle", "PostgreSQL", "SQL Server", "database performance tuning",
              "backup/recovery", "replication"]},
     [("Financial Services", "Senior DBA", "2014-01", "Present",
       "Manages 50+ production databases. No development or ML experience. Purely database administration.",
       ["Oracle", "PostgreSQL", "SQL Server"])],
     ("Coventry University", "B.Sc.", "Information Systems", "2009", "2013", "2:2"),
     10),

    ("gary_lewis", "Gary Lewis", 4,
     "iOS mobile developer with 6 years. Swift and SwiftUI expert. No backend, no Python, no ML experience.",
     {"Mobile": ["Swift", "SwiftUI", "UIKit", "Xcode", "Core Data", "App Store"]},
     [("Mobile Agency", "Senior iOS Developer", "2018-01", "Present",
       "Builds consumer iOS apps. Purely iOS development. No backend, ML, or Python experience.",
       ["Swift", "SwiftUI", "Xcode"])],
     ("Northumbria University", "B.Sc.", "Mobile Computing", "2014", "2018", "2:1"),
     6),

    ("patricia_hall", "Patricia Hall", 4,
     "UX/UI designer with 8 years. Expert in Figma, user research, and design systems. No technical development or ML skills.",
     {"Design": ["Figma", "Sketch", "Adobe XD", "user research", "prototyping",
                 "design systems", "accessibility"]},
     [("Design Studio", "Senior UX Designer", "2016-01", "Present",
       "Leads UX for enterprise SaaS products. No coding, ML, or AI technical skills.",
       ["Figma", "user research", "prototyping"])],
     ("Arts University Bournemouth", "B.A.", "Graphic Design", "2012", "2016", "First"),
     8),

    ("linda_wilson", "Linda Wilson", 4,
     "Technical project manager with 12 years. PMP certified. Manages engineering teams but no hands-on development or ML skills.",
     {"Management": ["project management", "Agile", "Scrum", "JIRA", "stakeholder management",
                     "risk management", "PMP", "budgeting"]},
     [("Enterprise Co.", "Senior Technical PM", "2012-01", "Present",
       "Manages 5 engineering teams. PMP certified, Scrum Master. No technical development or ML background.",
       ["JIRA", "Agile", "Scrum", "stakeholder management"])],
     ("Aston University", "MBA", "Business Administration", "2009", "2011", "Merit"),
     12),

    ("paul_robinson", "Paul Robinson", 4,
     "Network and security engineer with 10 years. CCNA/CCNP certified. No software development or ML background.",
     {"Networking": ["Cisco", "BGP", "OSPF", "network security", "firewalls",
                     "VPN", "CCNP", "Wireshark"]},
     [("ISP", "Senior Network Engineer", "2014-01", "Present",
       "Manages core routing infrastructure. Cisco certified. No programming, ML, or AI experience.",
       ["Cisco", "BGP", "network security"])],
     ("Staffordshire University", "HND", "Network Engineering", "2010", "2012", "Merit"),
     10),

    ("nancy_garcia", "Nancy Garcia", 4,
     "Business analyst with 8 years in requirements gathering and process improvement. No technical development or ML skills.",
     {"Business Analysis": ["requirements", "process mapping", "BPMN", "stakeholder interviews",
                             "BRD writing", "UAT", "JIRA"]},
     [("Consultancy", "Senior Business Analyst", "2016-01", "Present",
       "Gathers requirements and writes BRDs for software projects. No technical development background.",
       ["requirements gathering", "process mapping", "JIRA"])],
     ("Cardiff University", "B.A.", "Business Management", "2012", "2016", "2:1"),
     8),
]


def build_resume(seed: tuple) -> tuple[str, Resume]:
    """Build a Resume dataclass directly from seed data."""
    from schemas import ContactInfo, Skill, Experience, Education

    (cid, name, tier, summary, skills_dict, exp_list, edu_tuple, years) = seed

    # Contact
    first, *rest = name.lower().split()
    last = rest[-1] if rest else first
    contact = ContactInfo(
        name=name,
        email=f"{first}.{last}@email.com",
        phone=f"+44 7{cid.__hash__() % 900000000 + 100000000}",
        linkedin=f"linkedin.com/in/{first}-{last}",
        github=f"github.com/{first}{last}" if tier <= 2 else "",
    )

    # Skills
    skills = [Skill(category=cat, skills=sk) for cat, sk in skills_dict.items()]

    # Experience
    experience = []
    for (company, position, start, end, desc, techs) in exp_list:
        experience.append(Experience(
            company=company, position=position,
            start_date=start, end_date=end,
            description=desc,
            technologies=techs,
            achievements=[],
        ))

    # Education
    (inst, degree, field, start_e, end_e, gpa) = edu_tuple
    education = [Education(
        institution=inst, degree=degree,
        field_of_study=field,
        start_date=start_e, end_date=end_e,
        gpa=gpa,
    )]

    return cid, Resume(
        contact_info=contact,
        professional_summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        certifications=[], projects=[], languages=[],
        publications=[], volunteer_experience=[],
        awards_recognition=[], metadata={},
    )


def main():
    print(f"Seeding {len(SEEDS)} candidates into ChromaDB...\n")

    store = CandidateVectorStore(persist_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    existing = store.count()
    print(f"Current ChromaDB pool size: {existing}")

    # MongoDB (optional — skip gracefully if not running)
    try:
        from database import save_candidate, is_connected
        mongo_ok = is_connected()
        print(f"MongoDB: {'connected' if mongo_ok else 'not available — skipping'}")
    except Exception:
        mongo_ok = False
        save_candidate = None

    seeded, skipped = 0, 0
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for seed in SEEDS:
        cid = seed[0]
        tier = seed[2]
        try:
            candidate_id, resume = build_resume(seed)
            store.add_candidate(resume, candidate_id)
            if mongo_ok and save_candidate:
                save_candidate(candidate_id, resume.dict(), source="seeded")
            print(f"  [T{tier}] {resume.contact_info.name} ({candidate_id})")
            tier_counts[tier] += 1
            seeded += 1
        except Exception as e:
            print(f"  ERROR seeding {cid}: {e}")
            skipped += 1

    print(f"\nDone. Seeded: {seeded}  Skipped: {skipped}")
    print(f"Tier breakdown: {tier_counts}")
    print(f"Total pool size: {store.count()}")
    print("\nRun python interview_app.py and open http://localhost:8001 to demo.")


if __name__ == "__main__":
    main()
