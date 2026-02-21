# 🏆 AI Skill Gap Reasoning System  
## Hybrid Semantic + Generative AI Career Readiness System

---

## 📌 Project Overview

AI Skill Gap Reasoning System is a hybrid Artificial Intelligence system designed to evaluate a candidate’s readiness for a specific job role. The system analyzes a resume and job description, identifies skill gaps using semantic embeddings, and generates grounded, explainable career insights using Retrieval-Augmented Generation (RAG).

Unlike traditional resume screening systems that rely purely on keyword matching, this project integrates semantic similarity modeling and generative AI to provide structured, interpretable, and role-aware feedback.

---

# 📚 Background Study

Traditional resume screening systems such as Applicant Tracking Systems (ATS) primarily rely on keyword-based matching techniques. These systems compare exact words between resumes and job descriptions. While computationally efficient, keyword-based approaches fail to capture semantic similarity between related skill expressions. For example, “REST API development” and “backend service implementation” may describe similar competencies but may not match in strict keyword filtering systems.

Recent advancements in transformer-based Natural Language Processing (NLP) models have enabled semantic representation of textual data using embeddings. Sentence-level transformer models convert text into dense vector representations that capture contextual meaning instead of surface-level keywords. By computing cosine similarity between embeddings, semantic closeness between resume content and job requirements can be measured effectively.

Large Language Models (LLMs) further enhance intelligent systems by generating explainable feedback and personalized recommendations. However, direct LLM usage without structured grounding may lead to generic or unverified outputs (hallucinations). Retrieval-Augmented Generation (RAG) mitigates this issue by retrieving relevant domain-specific knowledge before response generation. This grounding mechanism improves reliability, contextual accuracy, and explainability.

This project integrates:

- Embedding-based semantic gap analysis  
- Cosine similarity-based reasoning  
- Retrieval-based knowledge grounding  
- Generative AI-based explanation  

The result is a hybrid, explainable career readiness assessment system.

---

# 📂 Dataset Description

This project uses publicly available datasets strictly for **evaluation and validation purposes only**.  
No supervised training or fine-tuning is performed on these datasets.

---

## 📝 Resume Dataset (PDF Format)

**Source Repository:**  
https://github.com/abhay-glitch/Resume-dataset-/tree/main/Resume%20in%20PDF

### Dataset Characteristics
- Real-world resume PDFs  
- Diverse candidate profiles  
- Structured sections (education, skills, experience)  
- Multiple professional domains  

### Purpose in This Project
- Validate PDF parsing and preprocessing pipeline  
- Evaluate semantic skill extraction robustness  
- Simulate real-world resume screening conditions  
- Support cross-domain evaluation  

---

## 💼 Job Description Dataset

**Source:**  
https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english

### Dataset Characteristics
- Large-scale English job descriptions  
- Multi-domain coverage  
- Rich requirement and skill descriptions  
- Real recruitment data structure  

### Purpose in This Project
- Provide diverse job role requirements  
- Enable cross-role skill comparison  
- Build retrieval knowledge base for RAG  
- Support evaluation across multiple domains  

---

# 🏗️ System Architecture

The system consists of three primary components:

## 1️⃣ Preprocessing Layer
- Resume PDF → Text extraction  
- Text cleaning and normalization  
- Sentence segmentation  

## 2️⃣ Semantic Skill Gap Engine
- Sentence Transformer embeddings  
- Predefined skill vocabulary  
- Cosine similarity computation  
- Classification into:
  - Matched Skills  
  - Missing Skills  
  - Partial Matches  

This layer performs structured reasoning without using a generative model.

## 3️⃣ RAG-Based Generative Layer
- Embedding-based knowledge retrieval  
- Context grounding  
- LLM-generated explanations  
- Personalized learning roadmap generation  

This layer represents the Generative AI component of the system.

---

# 🔬 Methodology

1. Parse resume PDF into text.  
2. Clean and normalize resume and job description.  
3. Convert text into semantic embeddings.  
4. Compare resume skills with job requirements using cosine similarity.  
5. Identify skill gaps.  
6. Retrieve relevant contextual knowledge (RAG).  
7. Generate grounded career insights and recommendations.  

---

# ⚙️ Technologies Used

- Python  
- Sentence Transformers (MiniLM)  
- Scikit-learn (Cosine Similarity)  
- Retrieval-Augmented Generation (RAG)  
- Large Language Model APIs  
- PDFPlumber  

---

# 📊 System Output

The system generates:

- Role match score  
- Identified skill gaps  
- Contextual explanation of missing skills  
- Personalized learning roadmap  
- Suggested project ideas  
- Career readiness summary  

---

# 🎓 Academic Contribution

This project demonstrates:

- Practical application of transformer-based embeddings  
- Semantic similarity modeling  
- Hybrid AI system architecture  
- Retrieval-Augmented Generation  
- Explainable AI design principles  

The focus is on intelligent system integration rather than model training, aligning with modern AI engineering practices.

---

# 🚀 Future Enhancements

- Multi-agent LLM orchestration  
- Domain-specific skill graphs  
- Advanced quantitative evaluation metrics  
- Recruiter-side analytics dashboard  
- Fine-tuned domain embeddings  

---

# 👨‍💻 Project Type

B.Tech – 3rd Year  
Generative AI Course Project  
Department of Computer Science