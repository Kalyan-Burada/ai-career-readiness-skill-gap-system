"""
FastAPI Backend for Career Readiness System
Provides REST API endpoints for skill analysis and career advice.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

# Import our modules
from resume_parser import extract_resume_text
from text_cleaner import clean_text
from text_utils import split_into_sentences
from embedding_module import generate_embeddings
from similarity_engine import compute_similarity_matrix, classify_gaps
from phrase_extracter import extract_candidate_phrases
from rag_engine import get_rag_engine
from llm_client import get_llm_client


app = FastAPI(
    title="Career Readiness AI API",
    description="AI-powered skill gap analysis and career guidance system",
    version="1.0.0"
)


def _filter_non_skills(skills: list) -> list:
    """
    Post-extraction safety filter: remove obviously non-skill terms
    that NLP extraction may have missed, and split comma-separated
    tool lists into individual skills.
    """
    import re as _re
    
    # First pass: split comma-separated entries into individual skills
    # e.g. "tableau, power bi" → ["tableau", "power bi"]
    expanded = []
    for skill in skills:
        if ',' in skill:
            parts = [p.strip() for p in skill.split(',') if p.strip()]
            expanded.extend(parts)
        else:
            expanded.append(skill)
    
    # Patterns that indicate a term is NOT a skill
    _reject_patterns = [
        r'^full[\s-]?time$', r'^part[\s-]?time$', r'^remote$', r'^hybrid$',
        r'^on[\s-]?site$', r'^contract$', r'^permanent$', r'^temporary$',
        # Education / degree terms
        r'^(bachelor|master|phd|doctorate|associate|mba)\b',
        r'\b(degree|diploma|certificate)\s*$',
        # Infinitive phrases ("to present insights", "to drive growth")
        r'^to\s+(present|drive|ensure|support|deliver|build|create|develop|implement|maintain|manage|lead|work)\b',
        # Task descriptions (verb-led phrases)
        r'^(build|create|develop|implement|translate|write|maintain|ensure|deliver|manage|lead|drive|coordinate|present|monitor|analyze|prepare|conduct|negotiate|solicit|showcase|entertain)\s',
        # Quality attributes without a technical domain qualifier
        r'^(new|existing|modern|various|multiple|seamless|reusable|clean|scalable|maintainable)\s+(code|features|functionality|components|applications|solutions|interfaces|systems|designs|updates|issues)$',
        # Generic descriptors
        r'^(technical|business|functional)\s+(feasibility|requirements|specifications|documentation|logic)$',
        r'^(strong|good|deep|solid|excellent|proven)\s+(experience|understanding|knowledge|background|skills|portfolio|track record|proficiency|negotiation|communication)$',
        # Employment / HR terms
        r'(salary|benefits|compensation|equal opportunity|employer)',
        # JD section headers
        r'^(job|position|role)\s+(summary|description|overview|posting|details)$',
        r'^(key|core|main|primary|your)\s+responsibilities$',
        r'^(minimum|preferred|required)\s+qualifications$',
        r'^about\s+(the|us|this)',
        r'^(who we are|what we offer|what you)',
        # Job platform references
        r'^(indeed|glassdoor|linkedin)\s*(jobs?)?$',
        r'^(job|career)\s+(board|page|site)s?$',
        r'^(remote|flexible)\s+work',
        r'^work\s+(options|arrangements|from home)$',
        # Generic problem / context descriptions
        r'^(complex|real[\s-]?world|diverse|various)\s+(problems?|sources?|data)$',
        r'^(production|product)\s+(systems?|features?|environments?)$',
        r'^(related|relevant|similar|quantitative)\s+(field|discipline)s?$',
        r'^(actionable|meaningful|valuable|data[\s-]?driven)\s+insights?$',
        r'^(cross[\s-]?functional|continuous|process)\s+(collaboration|improvement)$',
        r'^(competitive|strategic|informed)\s+(advantage|decisions?)$',
        r'^(day[\s-]?to[\s-]?day|daily|business)\s+operations$',
        # Single generic words that might slip through
        r'^(mathematics|indeed|diverse|key|production|operations|insights|collaboration|present|stakeholders|leadership)$',
        
        # ── Business / Sales / Hospitality non-skill patterns ──
        # Job titles
        r'^(business\s+travel\s+)?sales\s+manager$',
        r'^(inside\s+)?sales\s+(representative|associate|executive)$',
        r'^(account|business\s+development)\s+(manager|executive|representative)$',
        r'^(hotel|front\s+desk|revenue|regional|area|national)\s+manager$',
        # Business entity types (not skills)
        r'^(existing|new|high-value|corporate|strategic|key)\s+(clients?|accounts?|business)$',
        r'^(new\s+and\s+existing|existing\s+high-value)\s+(corporate\s+)?(clients?|accounts?)$',
        r'^(travel\s+management\s+companies|tmcs?|travel\s+agencies?)$',
        r'^(new\s+)?b2b\s+(business|clients?)$',
        # Business activities / deliverables (not skills)
        r'^(action|sales|business|marketing)\s+(plans?|blitzes?|strategy|strategies)$',
        r'^(annual|negotiated|corporate|commercial)\s+(rates?|terms?|contracts?)$',
        r'^(corporate\s+)?room\s+nights?$',
        r'^(monthly|weekly|quarterly|annual)[/\s]*(monthly|weekly|quarterly)?\s*(revenue\s+)?goals?$',
        r'^(weekly|monthly|quarterly)[/\s]*(monthly|weekly|quarterly)?\s*sales?\s*reports?$',
        r'^(property|site)\s+(inspections?|tours?)$',
        r'^(industry|networking|trade)\s+(events?|shows?)$',
        r'^(sales|marketing|outreach|business)\s+efforts?$',
        r'^(client|customer|account)\s+(relations?|relationships?|retention|acquisition)$',
        r'^(competitor|competitive|market)\s+(activity|analysis|trends?|research)$',
        r'^(contract|rate)\s+negotiation$',
        r'^(lead|pipeline)\s+(generation|management)$',
        r'^(booking|travel\s+booking)\s+(process|errors?)$',
        r'^(customer|client)\s+(engagement|satisfaction)$',
        # Experience descriptors
        r'^\d[\d+\-]*\s*years?\b',
        r'^(specifically|actively)\s+targeting$',
        r'^(strong|excellent)\s+proficiency$',
    ]
    
    filtered = []
    seen = set()  # Deduplicate after splitting
    for skill in expanded:
        s_lower = skill.lower().strip()
        if not s_lower or len(s_lower) < 2:
            continue
        # Deduplicate
        if s_lower in seen:
            continue
        # Also check comma-stripped version
        s_no_comma = s_lower.replace(',', ' ').strip()
        s_no_comma = _re.sub(r'\s+', ' ', s_no_comma)
        is_rejected = False
        for pattern in _reject_patterns:
            if _re.search(pattern, s_lower) or _re.search(pattern, s_no_comma):
                is_rejected = True
                break
        if not is_rejected:
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


# Global instances (initialized on startup)
rag_engine = None
llm_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine and LLM client on startup."""
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
            if len(s.strip()) > 3
        ]
        
        clean_jd_sentences = [
            clean_text(s)
            for s in split_into_sentences(job_description)
            if len(s.strip()) > 0
        ]
        
        # Extract skills
        resume_skills = extract_candidate_phrases(clean_resume_sentences)
        jd_skills = extract_candidate_phrases(clean_jd_sentences)
        
        if not jd_skills or not resume_skills:
            raise HTTPException(
                status_code=400,
                detail="No skills extracted. Verify input quality."
            )
        
        # Step 3: Semantic Skill Gap Analysis
        resume_embeddings = generate_embeddings(resume_skills)
        jd_embeddings = generate_embeddings(jd_skills)
        
        similarity_matrix = compute_similarity_matrix(jd_embeddings, resume_embeddings)
        
        matched, missing = classify_gaps(jd_skills, resume_skills, similarity_matrix)
        
        # Post-filter: remove non-skill terms that slipped through NLP extraction
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
