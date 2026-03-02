"""
FastAPI Backend for Career Readiness System
Provides REST API endpoints for skill analysis and career advice.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from resume_parser import extract_resume_text
from text_cleaner import clean_text
from text_utils import split_into_sentences
from embedding_module import generate_embeddings
from similarity_engine import compute_similarity_matrix, classify_gaps
from phrase_extracter import extract_candidate_phrases
from rag_engine import get_rag_engine
from llm_client import get_llm_client


# Global instances
rag_engine = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global rag_engine, llm_client
    
    print("🚀 Starting Career Readiness API...")
    
    # Initialize RAG engine
    try:
        rag_engine = get_rag_engine()
        print("✓ RAG engine initialized")
    except Exception as e:
        print(f"⚠️  RAG engine initialization failed: {e}")
    
    # Initialize LLM client
    llm_client = get_llm_client()
    if llm_client:
        print(f"✓ LLM client initialized ({llm_client.provider})")
    else:
        print("⚠️  LLM client not available (API key missing)")
    
    print("✓ API ready!")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="Career Readiness AI API",
    description="AI-powered skill gap analysis and career guidance system",
    version="1.0.0",
    lifespan=lifespan
)


def _filter_non_skills(skills: list) -> list:
    """
    Post-extraction safety filter: split comma-separated tool lists into individual skills.
    """
    expanded = []
    for skill in skills:
        if ',' in skill:
            parts = [p.strip() for p in skill.split(',') if p.strip()]
            expanded.extend(parts)
        else:
            expanded.append(skill)

    filtered = []
    seen = set()  # Deduplicate after splitting
    for skill in expanded:
        s_lower = skill.lower().strip()
        if not s_lower or len(s_lower) < 2:
            continue
        if s_lower in seen:
            continue
        filtered.append(skill)
        seen.add(s_lower)
    return filtered

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class JobDescriptionRequest(BaseModel):
    job_description: str


class SkillAnalysisResponse(BaseModel):
    jd_skills_count: int
    resume_skills_count: int
    matched_skills: List[str]
    missing_skills: List[str]
    match_score: float
    career_advice: Optional[dict] = None


# CORS middleware


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Career Readiness AI API",
        "version": "1.0.0",
        "rag_available": rag_engine is not None,
        "llm_available": llm_client is not None
    }


@app.post("/api/analyze", response_model=SkillAnalysisResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = File(...)
):
    """
    Analyze resume against job description.
    
    Args:
        resume: PDF file upload
        job_description: Job description text
    
    Returns:
        Skill gap analysis with career advice
    """
    try:
        # Step 1: Resume Ingestion
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await resume.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        raw_resume = extract_resume_text(tmp_path)
        os.unlink(tmp_path)  # Clean up temp file
        
        if not raw_resume or len(raw_resume.strip()) < 50:
            raise HTTPException(status_code=400, detail="Failed to extract text from resume")
        
        # Step 2: Text Preprocessing
        clean_resume_sentences = [
            clean_text(s)
            for s in split_into_sentences(raw_resume)
            if len(s.strip()) >= 3
        ]
        
        clean_jd_sentences = [
            clean_text(s)
            for s in split_into_sentences(job_description)
            if len(s.strip()) >= 3
        ]
        
        # Extract skills & Perform strict zero-hallucination analysis
        if llm_client:
            print("🧠 Performing native side-by-side exact extraction and gap analysis using LLM...")
            analysis_dict = llm_client.perform_full_gap_analysis(raw_resume, job_description)
            
            if analysis_dict:
                jd_skills = analysis_dict.get("jd_skills", [])
                resume_skills = analysis_dict.get("resume_skills", [])
                matched = analysis_dict.get("matched_skills", [])
                missing = analysis_dict.get("missing_skills", [])
            else:
                resume_skills = extract_candidate_phrases(clean_resume_sentences)
                jd_skills = extract_candidate_phrases(clean_jd_sentences)
                resume_embeddings = generate_embeddings(resume_skills)
                jd_embeddings = generate_embeddings(jd_skills)
                similarity_matrix = compute_similarity_matrix(jd_embeddings, resume_embeddings)
                matched, missing = classify_gaps(jd_skills, resume_skills, similarity_matrix)
        else:
            resume_skills = extract_candidate_phrases(clean_resume_sentences)
            jd_skills = extract_candidate_phrases(clean_jd_sentences)
            resume_embeddings = generate_embeddings(resume_skills)
            jd_embeddings = generate_embeddings(jd_skills)
            similarity_matrix = compute_similarity_matrix(jd_embeddings, resume_embeddings)
            matched, missing = classify_gaps(jd_skills, resume_skills, similarity_matrix)

        if not jd_skills or not resume_skills:
            raise HTTPException(
                status_code=400,
                detail="No skills extracted. Verify input quality."
            )

        missing = _filter_non_skills(missing)
        
        match_score = (len(matched) / len(jd_skills) * 100) if jd_skills else 0
        
        # Step 4 & 5: RAG + LLM for career advice
        career_advice = None
        if rag_engine and llm_client and missing:
            try:
                # Retrieve context for missing skills
                skill_contexts = rag_engine.get_context_for_missing_skills(missing[:5])
                
                # Generate career advice using LLM (pass full JD for domain awareness)
                career_advice = llm_client.generate_career_advice(
                    matched_skills=matched,
                    missing_skills=missing,
                    skill_contexts=skill_contexts,
                    job_description=job_description
                )
            except Exception as e:
                print(f"Career advice generation failed: {e}")
                career_advice = {"error": str(e)}
        
        return SkillAnalysisResponse(
            jd_skills_count=len(jd_skills),
            resume_skills_count=len(resume_skills),
            matched_skills=sorted(matched),
            missing_skills=sorted(missing),
            match_score=round(match_score, 1),
            career_advice=career_advice
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/skill-context")
async def get_skill_context(skill: str):
    """Get learning resources and context for a specific skill."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    
    try:
        context = rag_engine.get_context_for_skill(skill)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "api": "healthy",
        "rag_engine": "available" if rag_engine else "unavailable",
        "llm_client": "available" if llm_client else "unavailable",
        "llm_provider": llm_client.provider if llm_client else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
