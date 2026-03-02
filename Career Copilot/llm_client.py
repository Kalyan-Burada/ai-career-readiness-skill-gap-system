"""
LLM Integration Module
Supports both OpenAI GPT and Google Gemini APIs for career advice generation.
"""

import os
import json
from typing import List, Dict
import re
from pydantic import BaseModel

class SkillExtraction(BaseModel):
    technical_skills: List[str]
    soft_skills: List[str]
    tools_and_frameworks: List[str]
    domain_expertise: List[str]

class JobContext(BaseModel):
    job_title: str
    domain: str


class CareerPathSchema(BaseModel):
    immediate: List[str]
    after_upskilling: List[str]

class ActionPlanSchema(BaseModel):
    weeks_1_4: List[str]
    weeks_5_8: List[str]
    weeks_9_12: List[str]

class PrioritySkillSchema(BaseModel):
    skill: str
    reason: str
    timeline: str
    actions: List[str]

class ProjectSchema(BaseModel):
    name: str
    description: str
    intuition: str
    tech_stack: str
    skills_covered: List[str]

class CareerAdviceSchema(BaseModel):
    career_summary: str
    strengths: List[str]
    priority_skills: List[PrioritySkillSchema]
    recommended_projects: List[ProjectSchema]
    career_paths: CareerPathSchema
    action_plan: ActionPlanSchema

class LLMClient:
    """Unified interface for LLM providers."""
    
    def __init__(self, provider="openai", api_key=None):
        """
        Initialize LLM client.
        
        Args:
            provider: "openai" or "gemini"
            api_key: API key (reads from env if not provided)
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _get_api_key(self):
        """Get API key from environment variables."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-4o-mini"  # Cost-effective model
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_gemini(self):
        """Initialize Gemini client."""
        try:
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
            self.model = "gemini-2.0-flash"  # Updated model name
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Run: pip install google-genai")

    def extract_skills_dynamically(self, text: str) -> SkillExtraction:
        """
        Dynamically extract ONLY valid professional skills and domain expertise concepts from raw text.
        Pro-Tip: Multi-Stage Extraction using Pydantic Schema and Gemini Structured Outputs.
        """
        if self.provider == "gemini":
            # Stage 1: Broad Extraction
            stage_1_prompt = f"Extract every single possible professional skill, tool, framework, and domain expertise from this text into a raw comma-separated list:\n\n{text[:6000]}"
            try:
                stage_1_response = self.client.models.generate_content(
                    model=self.model,
                    contents=stage_1_prompt,
                )
                raw_skills = stage_1_response.text.strip()
            except Exception as e:
                print(f"LLM Stage 1 extraction failed: {e}")
                raw_skills = text[:6000] # Fallback to original text if extraction fails
                
            # Stage 2: Validation & Structured Categorization
            stage_2_prompt = (
                f"Here is a raw list of extracted skills:\n{raw_skills}\n\n"
                "Filter this list. Remove generic words (e.g., 'teamwork', 'results', 'experience', 'Monday') "
                "and keep only industry-standard technical competencies, tools, professional soft skills, and domain expertise. "
                "Categorize them accurately."
            )
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=stage_2_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": SkillExtraction,
                        "temperature": 0.0
                    }
                )
                return response.parsed
            except Exception as e:
                print(f"LLM Stage 2 structured extraction failed: {e}")
                return SkillExtraction(technical_skills=[], soft_skills=[], tools_and_frameworks=[], domain_expertise=[])
                
        elif self.provider == "openai":
            prompt = (
                "You are an expert ATS skill extractor. Extract all professional skills from this text and categorize them "
                "strictly into: technical_skills, soft_skills, tools_and_frameworks, domain_expertise.\n\n"
                f"TEXT TO PARSE:\n{text[:6000]}"
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return ONLY a JSON object matching the requested categories as arrays of strings."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" },
                    temperature=0.0
                )
                parsed_json = json.loads(response.choices[0].message.content.strip())
                return SkillExtraction(
                    technical_skills=parsed_json.get("technical_skills", []),
                    soft_skills=parsed_json.get("soft_skills", []),
                    tools_and_frameworks=parsed_json.get("tools_and_frameworks", []),
                    domain_expertise=parsed_json.get("domain_expertise", [])
                )
            except Exception as e:
                print(f"LLM OpenAI extraction failed: {e}")
                return SkillExtraction(technical_skills=[], soft_skills=[], tools_and_frameworks=[], domain_expertise=[])

    def extract_skills(self, text: str) -> List[str]:
        """
        Legacy wrapper for backwards compatibility. Returns flattened list of all skills.
        """
        parsed = self.extract_skills_dynamically(text)
        return list(set(
            parsed.technical_skills + 
            parsed.soft_skills + 
            parsed.tools_and_frameworks + 
            parsed.domain_expertise
        ))

    def perform_full_gap_analysis(self, resume_text: str, jd_text: str) -> Dict:
        """
        The absolute 'best ever' dynamic analysis requested by the user:
        Extracts directly from resume and job description, compares them side-by-side natively
        without blindly relying on pre-existing lists or weak cosine similarity.
        """
        prompt = (
            "You are a highly advanced Applicant Tracking System and Technical Recruiter. "
            "Your task is to perform an absolutely flawless skill gap analysis between the provided Resume and Job Description. "
            "You must cross-reference them and figure out what required skills are present and missing.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY extract hard skills, tools, technologies, methodologies, and clear domain competencies.\n"
            "2. DO NOT add, guess, or hallucinate any skill that is not explicitly in the text.\n"
            "3. DO NOT output any markdown, codeblocks, or extra text. ONLY output raw JSON.\n\n"
            "Respond EXACTLY in this JSON format strictly:\n"
            "{\n"
            '  "jd_skills": ["List", "of", "all", "required", "skills", "from", "JD"],\n'
            '  "resume_skills": ["List", "of", "candidate", "skills", "from", "Resume"],\n'
            '  "matched_skills": ["Required", "skills", "that", "the", "candidate", "HAS"],\n'
            '  "missing_skills": ["Required", "skills", "that", "the", "candidate", "is", "MISSING"]\n'
            "}\n\n"
            f"--- RESUME ---\n{resume_text[:4000]}\n\n"
            f"--- JOB DESCRIPTION ---\n{jd_text[:4000]}"
        )

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional ATS skill extractor. Return ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={ "type": "json_object" }
                )
                result_text = response.choices[0].message.content.strip()

            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                result_text = response.text.replace("```json", "").replace("```", "").strip()
            
            # Parse the strict JSON
            return json.loads(result_text)
            
        except Exception as e:
            print(f"LLM native gap analysis failed: {e}")
            return None

    def generate_career_advice(
        self, 
        matched_skills: List[str],
        missing_skills: List[str],
        skill_contexts: List[Dict],
        job_description: str = ""
    ) -> Dict:
        """
        Generate comprehensive career advice using LLM Dynamically with Structured Schemas.
        """
        job_title, domain = self._detect_job_context(job_description)
        
        prompt = self._build_career_advice_prompt(
            matched_skills, missing_skills, skill_contexts,
            job_title=job_title, domain=domain, job_description=job_description
        )

        try:
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": CareerAdviceSchema,
                        "temperature": 0.2
                    }
                )
                return response.parsed.model_dump()
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a senior career advisor. Return ONLY valid JSON matching the exact expected schema with keys: career_summary, strengths, priority_skills, recommended_projects, career_paths, action_plan."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"LLM roadmap generation failed: {e}")
            return {
                "career_summary": "We couldn't generate a dynamic summary right now. Focus on the missing skills identified.",
                "strengths": matched_skills[:5],
                "priority_skills": [{"skill": s, "reason": "Required by JD", "timeline": "2-4 weeks", "actions": ["Find online tutorials", "Build a small project"]} for s in missing_skills[:3]],
                "recommended_projects": [
                    {
                        "name": "Skill Integration Project",
                        "description": "Integrate the missing skills into a portfolio project.",
                        "intuition": "Hands-on practice is the best way to learn.",
                        "tech_stack": ", ".join(missing_skills[:3]),
                        "skills_covered": missing_skills[:3]
                    }
                ],
                "career_paths": {"immediate": [job_title], "after_upskilling": [f"Senior {job_title}"]},
                "action_plan": {
                    "weeks_1_4": ["Focus on learning the fundamentals of the missing skills."],
                    "weeks_5_8": ["Build hands-on projects and apply your knowledge."],
                    "weeks_9_12": ["Polish resume and perform mock interviews."]
                }
            }

    def _detect_job_context(self, job_description: str):
        """
        Dynamically extract job title and domain from the job description text using LLM.
        Returns (job_title, domain) tuple.
        """
        if not job_description or len(job_description.strip()) < 10:
            return "Professional", "general"
            
        prompt = (
            "Analyze the following job description and extract the precise 'job_title' "
            "and categorize the broad industry 'domain' (e.g., frontend, backend, sales, finance, healthcare, data_science, general).\n\n"
            f"JOB DESCRIPTION:\n{job_description[:4000]}"
        )
        
        try:
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": JobContext,
                        "temperature": 0.0
                    }
                )
                parsed = response.parsed
                return parsed.job_title, parsed.domain
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an ATS parser. Return a JSON object with 'job_title' and 'domain' as string fields."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                parsed_json = json.loads(response.choices[0].message.content.strip())
                return parsed_json.get("job_title", "Professional"), parsed_json.get("domain", "general")
        except Exception as e:
            print(f"LLM Job Context extraction failed: {e}")
            
        return "Professional", "general"
    
    def _build_career_advice_prompt(self, matched_skills, missing_skills, skill_contexts,
                                     job_title="", domain="", job_description=""):
        """Build comprehensive prompt for career advice generation."""
        prompt = f"""You are an expert career coach and AI-powered career advisor. Analyze the following skill gap assessment and provide actionable career guidance.

**TARGET ROLE:** {job_title if job_title else 'Not specified'}
**DOMAIN:** {domain if domain else 'general'}

**JOB DESCRIPTION EXCERPT:**
{job_description[:800] if job_description else 'Not provided'}

**MATCHED SKILLS ({len(matched_skills)}):**
{', '.join(matched_skills[:15])}{'...' if len(matched_skills) > 15 else ''}

**MISSING SKILLS ({len(missing_skills)}):**
{', '.join(missing_skills)}

**SKILL DEVELOPMENT CONTEXT:**
"""
        
        for ctx in skill_contexts[:5]:  # Limit to top 5 for token efficiency
            prompt += f"\n{ctx['skill'].upper()}:\n"
            prompt += f"- Description: {ctx['description']}\n"
            prompt += f"- Time to learn: {ctx['estimated_time']}\n"
            prompt += f"- Learning resources: {', '.join(ctx['learning_resources'][:2])}\n"
        
        prompt += """

**YOUR TASK:**
Act as a world-class principal career mentor and technical architect for top-tier tech companies. Generate an elite, highly-technical, and hyper-personalized career roadmap based strictly on the user's specific skill gap data. 

Deliver your guidance in a professional, decisive, yet deeply insightful tone. Do not use generic filler words.

1. **Career Readiness Summary** (2-3 sentences max):
   - Provide an unfiltered, executive-level diagnostic of where the candidate stands in the current market based *only* on matched vs. missing skills.

2. **Priority Skill Development Plan** (Top 3-5 missing skills):
   - CRITICAL: Each skill's "reason" field MUST be completely unique and explain the specific engineering or business value of THAT particular skill. NEVER use a generic reason like "This skill is valuable for professional growth" — that is UNACCEPTABLE.
   - CRITICAL: Each skill's "actions" list MUST contain 3-5 hyper-specific steps unique to THAT skill. NEVER use "Search for online courses" or "Read industry-specific books" — those are GENERIC GARBAGE. Instead, name EXACT courses, books, tools, and certifications.
   - Each skill's "timeline" MUST be specific to that skill's complexity (e.g., "3-4 weeks" for a tool, "2-3 months" for a framework, "4-6 months" for a paradigm).

   **EXAMPLE of EXCELLENT priority_skills output (use this quality standard):**
   ```
   {{"skill": "interactive dashboards", "reason": "Modern data-driven organizations require real-time visual analytics. Interactive dashboards built with tools like Tableau, Power BI, or Plotly Dash enable stakeholders to explore KPIs, filter segments, and drill down into metrics without engineering support — a critical differentiator for any analytics or PM role.", "timeline": "4-6 weeks with dedicated practice", "actions": ["Master Tableau Desktop: complete Tableau's free e-learning on calculated fields, LOD expressions, and dashboard actions", "Build 3 interactive dashboards using real datasets from Kaggle (sales, HR analytics, supply chain)", "Learn Power BI DAX formulas and create parameterized reports with slicers and bookmarks", "Study dashboard UX principles: Storytelling with Data by Cole Nussbaumer Knaflic", "Publish dashboards to Tableau Public and embed in your portfolio site"]}}
   ```
   
   **EXAMPLE of TERRIBLE output (NEVER do this):**
   ```
   {{"skill": "interactive dashboards", "reason": "This skill is valuable for professional growth and career advancement.", "timeline": "Varies by skill complexity (3-12 months)", "actions": ["Search for online courses on Coursera, Udemy, or LinkedIn Learning", "Read industry-specific books and blogs", "Join professional communities and attend conferences"]}}
   ```

3. **Elite Recommended Projects** (2-3 portfolio projects):
   - Project Name: Must sound like a genuine enterprise-grade initiative (e.g., "Real-time Supply Chain Optimization Engine" or "Distributed AI Model Monitoring System").
   - Project Intuition: Explain the *why*. What exact business problem are you solving? Who are the stakeholders? How does this bridge the candidate's exact missing skills?
   - Deliverables/Architecture: Mention specific tech stacks (e.g., matching the missing skills), architectures, and how success is measured.

4. **Career Path Recommendations**:
   - Provide highly specific titles (e.g., MLOps Engineer, Technical Product Manager - Data Infrastructure) for immediately matchable roles, and where they can pivot *after* their upskilling.

5. **Elite 90-Day Action Plan** (The Ultimate Blueprint):
   - **Weeks 1-4 (Deep Learning Phase):** Name the explicit sub-topics, algorithms, or frameworks the user must study for their missing skills. What concepts must they concentrate deeply on?
   - **Weeks 5-8 (Execution & Practice Phase):** Describe explicit coding, architecture, or business strategies they must practice. Instruct them to build the exact projects you recommended, outlining what specific features to code.
   - **Weeks 9-12 (Polish & Market Entry):** Instructions on how to frame these new skills on a resume and specific interview paradigms (e.g., System Design, Product Sense) they need to prep based on their profile.

Format your response as JSON with these exact keys:
{
  "career_summary": "string",
  "strengths": ["string"],
  "priority_skills": [{"skill": "string", "reason": "string", "timeline": "string", "actions": ["string"]}],
  "recommended_projects": [{"name": "string", "description": "string", "intuition": "string", "tech_stack": "string", "skills_covered": ["string"]}],
  "career_paths": {"immediate": ["string"], "after_upskilling": ["string"]},
  "action_plan": {"weeks_1_4": ["string"], "weeks_5_8": ["string"], "weeks_9_12": ["string"]}
}

CRITICAL RULES:
- Every priority skill's "reason" MUST be a unique 2-3 sentence explanation of that skill's specific technical/business value. NO TWO SKILLS should have the same reason.
- Every priority skill's "actions" MUST list 3-5 SPECIFIC steps with named resources (exact course names, book titles, tool names, certification names). NO GENERIC "search for courses" or "read blogs".
- Every priority skill's "timeline" MUST be a specific timeframe for that exact skill (e.g., "3-4 weeks", "6-8 weeks"), NOT "Varies by skill complexity".
- Every project MUST have an enterprise-grade sounding name relevant to the "{domain}" domain.
- Every project MUST have a detailed "intuition" field explaining the *business problem* it solves and *why* it bridges the candidate's skill gaps.
- Every project MUST have a "tech_stack" field listing exact technologies.
- The 90-day plan MUST name specific topics, sub-topics, tools, algorithms, or frameworks to study — NOT generic advice.
- All recommendations MUST be specific to the "{domain}" domain and the "{job_title}" role.

Provide practical, specific advice. Be encouraging but realistic.
"""
        return prompt
    

def get_llm_client(provider=None):
    """
    Get LLM client instance.
    """
    if provider:
        try:
            return LLMClient(provider=provider)
        except Exception:
            pass
            
    try:
        if os.getenv("OPENAI_API_KEY"):
            return LLMClient(provider="openai")
        elif os.getenv("GEMINI_API_KEY"):
            return LLMClient(provider="gemini")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        
    return None
