"""
LLM Integration Module
Supports both OpenAI GPT and Google Gemini APIs for career advice generation.
"""

import os
import json
from typing import List, Dict
import re


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
    
    def generate_career_advice(
        self, 
        matched_skills: List[str],
        missing_skills: List[str],
        skill_contexts: List[Dict],
        job_description: str = ""
    ) -> Dict:
        """
        Generate comprehensive career advice using LLM.
        
        Args:
            matched_skills: Skills present in resume
            missing_skills: Skills missing from resume
            skill_contexts: RAG-retrieved context for missing skills
            job_description: Original job description text for domain context
        
        Returns:
            Dictionary with career summary, roadmap, and recommendations
        """
        # Detect job title and domain from JD
        job_title, domain = self._detect_job_context(job_description)
        
        prompt = self._build_career_advice_prompt(
            matched_skills, missing_skills, skill_contexts,
            job_title=job_title, domain=domain, job_description=job_description
        )

        try:
            if self.provider == "openai":
                result = self._generate_openai(prompt)
            else:
                result = self._generate_gemini(prompt)
            
            # Post-process: replace any generic/lazy LLM output with rich local guidance
            result = self._enforce_quality(result, missing_skills, job_title, domain)
            return result
        except Exception as e:
            return self._build_local_guidance(
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                skill_contexts=skill_contexts,
                error_msg=str(e),
                job_title=job_title,
                domain=domain
            )
    
    def _detect_job_context(self, job_description: str):
        """
        Extract job title and domain from the job description text.
        Returns (job_title, domain) tuple.
        """
        if not job_description:
            return "Professional", "general"
        
        jd_lower = job_description.lower()
        
        # Try to extract job title from common patterns
        job_title = None
        title_patterns = [
            r'job\s*title\s*[:：]\s*(.+?)(?:\n|$)',
            r'position\s*[:：]\s*(.+?)(?:\n|$)',
            r'role\s*[:：]\s*(.+?)(?:\n|$)',
            r'job\s*description\s*[:：]\s*(.+?)(?:\n|$)',
            r'(?:looking\s+for|hiring)\s+(?:an?\s+)?(?:experienced\s+)?(.+?)(?:\s+to\s+|\s+who\s+|\.\s*)',
            r'🔹\s*job\s*title\s*[:：]\s*(.+?)(?:\n|$)',
        ]
        for pattern in title_patterns:
            match = re.search(pattern, jd_lower)
            if match:
                extracted = match.group(1).strip().rstrip('.').strip()
                if 3 < len(extracted) < 80:
                    job_title = extracted.title()
                    break
        
        # Detect domain based on keywords
        domain_keywords = {
            "frontend": ["frontend", "front-end", "front end", "react", "angular", "vue",
                         "css", "html", "ui developer", "ui engineer", "web developer",
                         "javascript developer", "typescript"],
            "backend": ["backend", "back-end", "back end", "server-side", "api developer",
                        "node.js", "django", "flask", "fastapi", "spring boot", "microservices"],
            "fullstack": ["full-stack", "full stack", "fullstack"],
            "data_science": ["data scientist", "machine learning", "deep learning",
                            "nlp", "computer vision", "statistical modeling"],
            "data_engineering": ["data engineer", "etl", "data pipeline", "spark",
                                "airflow", "data warehouse", "databricks"],
            "devops": ["devops", "site reliability", "sre", "infrastructure",
                      "kubernetes", "terraform", "ci/cd", "platform engineer"],
            "cloud": ["cloud engineer", "cloud architect", "aws", "azure", "gcp",
                     "solutions architect"],
            "product_management": ["product manager", "product management", "product owner",
                                  "product lifecycle", "roadmap", "stakeholder"],
            "data_analytics": ["data analyst", "business analyst", "analytics",
                              "power bi", "tableau", "reporting", "dashboard"],
            "mobile": ["mobile developer", "ios", "android", "react native",
                      "flutter", "swift", "kotlin"],
            "ai_ml": ["ai engineer", "ml engineer", "mlops", "artificial intelligence",
                      "ai product", "ai-driven", "ai-powered"],
            "security": ["security engineer", "cybersecurity", "penetration testing",
                        "security analyst", "infosec"],
            "qa": ["qa engineer", "quality assurance", "test automation", "sdet",
                   "testing engineer"],
            # ── Non-tech domains ──
            "sales": ["sales manager", "sales representative", "account executive",
                     "business development", "cold calling", "lead generation",
                     "account acquisition", "b2b sales", "b2b business",
                     "sales blitz", "sales strategy", "pipeline",
                     "crm", "salesforce", "hubspot", "pipedrive",
                     "inside sales", "outside sales", "field sales"],
            "hospitality": ["hotel sales", "hospitality", "room nights",
                           "corporate rates", "property site", "hotel services",
                           "travel management", "tmcs", "concierge",
                           "front desk", "guest services", "hotel management",
                           "amadeus", "opera pms", "sabre", "concur travel"],
            "marketing": ["marketing manager", "digital marketing", "seo", "sem",
                         "content marketing", "social media marketing", "brand manager",
                         "marketing automation", "marketo", "mailchimp",
                         "campaign management", "email marketing", "ppc"],
            "finance": ["financial analyst", "accountant", "finance manager",
                       "financial modeling", "budgeting", "forecasting",
                       "accounts payable", "accounts receivable", "bookkeeping",
                       "quickbooks", "sap", "financial reporting"],
            "hr": ["human resources", "hr manager", "recruiter", "talent acquisition",
                  "employee relations", "compensation", "payroll",
                  "hris", "workday", "bamboohr", "applicant tracking"],
            "healthcare": ["healthcare", "clinical", "patient care", "medical",
                          "nursing", "pharmacy", "ehr", "epic", "cerner",
                          "hipaa", "medical records", "telehealth"],
            "supply_chain": ["supply chain", "logistics", "procurement",
                            "inventory management", "warehouse", "distribution",
                            "demand planning", "sourcing", "vendor management"],
        }
        
        domain = "general"
        max_hits = 0
        for d, keywords in domain_keywords.items():
            hits = sum(1 for kw in keywords if kw in jd_lower)
            if hits > max_hits:
                max_hits = hits
                domain = d
        
        # If no title extracted, infer from domain
        if not job_title:
            domain_default_titles = {
                "frontend": "Frontend Developer",
                "backend": "Backend Developer",
                "fullstack": "Full Stack Developer",
                "data_science": "Data Scientist",
                "data_engineering": "Data Engineer",
                "devops": "DevOps Engineer",
                "cloud": "Cloud Engineer",
                "product_management": "Product Manager",
                "data_analytics": "Data Analyst",
                "mobile": "Mobile Developer",
                "ai_ml": "AI/ML Engineer",
                "security": "Security Engineer",
                "qa": "QA Engineer",
                "sales": "Sales Professional",
                "hospitality": "Hospitality Sales Manager",
                "marketing": "Marketing Professional",
                "finance": "Finance Professional",
                "hr": "HR Professional",
                "healthcare": "Healthcare Professional",
                "supply_chain": "Supply Chain Professional",
            }
            job_title = domain_default_titles.get(domain, "Professional")
        
        return job_title, domain
    
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
    
    def _generate_openai(self, prompt):
        """Generate response using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert career coach providing actionable guidance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result
    
    def _generate_gemini(self, prompt):
        """Generate response using Google Gemini."""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            raise ValueError("LLM response did not contain valid JSON.")
        result = json.loads(json_match.group())
        return result
    
    def _format_error_message(self, error_msg: str) -> str:
        lowered = error_msg.lower()
        if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:
            return (
                "Gemini quota exceeded. Generated a local fallback roadmap from your skill gaps. "
                "Retry in a minute, use a different Gemini key/project, or switch to OPENAI_API_KEY."
            )
        if "api key" in lowered or "unauthorized" in lowered or "permission" in lowered:
            return "LLM authentication failed. Generated a local fallback roadmap. Check your API key."
        return "LLM request failed. Generated a local fallback roadmap using your skill-gap analysis."

    def _enforce_quality(self, result: dict, missing_skills: list, job_title: str, domain: str) -> dict:
        """
        Post-process LLM output: detect and replace generic/lazy content
        with our rich skill-specific guidance. This ensures every skill
        gets unique, actionable advice even if the LLM produces boilerplate.
        Also validates the 90-day action plan is domain-relevant and specific.
        """
        # Phrases that indicate the LLM was lazy / generic
        _GENERIC_MARKERS = [
            "this skill is valuable",
            "valuable for professional growth",
            "career advancement",
            "professional development",
            "important skill to have",
            "important for career",
            "essential for success",
            "beneficial for your career",
            "highly sought after",
            "in-demand skill",
        ]
        _GENERIC_ACTION_MARKERS = [
            "search for online courses",
            "online courses on coursera",
            "read industry-specific books",
            "join professional communities",
            "attend conferences",
            "find mentors in your",
            "network with professionals",
            "take a course",
            "enroll in a course",
        ]

        # Non-skill terms that should never appear as priority skills
        _NON_SKILL_TERMS = {
            "bachelor s", "bachelors", "bachelor", "master s", "masters", "master",
            "phd", "mba", "doctorate", "degree", "to present insights",
            "present insights", "indeed jobs", "job summary", "full time",
            "full-time", "part-time", "remote work", "hybrid"
        }

        # ── Fix priority skills ──
        priority_skills = result.get("priority_skills", [])
        if priority_skills:
            improved_skills = []
            for skill_entry in priority_skills:
                skill_name = skill_entry.get("skill", "")
                
                # Skip non-skill entries entirely
                if skill_name.lower().strip() in _NON_SKILL_TERMS:
                    continue
                if re.match(r"^(bachelor|master|phd|doctorate|mba|associate)\b", skill_name.lower()):
                    continue
                
                reason = skill_entry.get("reason", "")
                actions = skill_entry.get("actions", [])
                timeline = skill_entry.get("timeline", "")

                reason_lower = reason.lower()
                actions_lower = " ".join(a.lower() for a in actions)
                timeline_lower = timeline.lower()

                reason_is_generic = any(m in reason_lower for m in _GENERIC_MARKERS) or len(reason) < 40
                generic_action_count = sum(1 for m in _GENERIC_ACTION_MARKERS if m in actions_lower)
                actions_are_generic = generic_action_count >= 1 or len(actions) < 2
                timeline_is_generic = "varies" in timeline_lower or "3-12" in timeline_lower or len(timeline) < 5

                if reason_is_generic or actions_are_generic or timeline_is_generic:
                    local = self._get_skill_specific_guidance(skill_name, domain, job_title)
                    if reason_is_generic:
                        skill_entry["reason"] = local["reason"]
                    if actions_are_generic:
                        skill_entry["actions"] = local["actions"]
                    if timeline_is_generic:
                        skill_entry["timeline"] = local["timeline"]

                improved_skills.append(skill_entry)
            result["priority_skills"] = improved_skills

        # ── Fix 90-Day Action Plan ──
        action_plan = result.get("action_plan", {})
        if action_plan:
            plan_text = " ".join(
                " ".join(action_plan.get(k, [])) for k in ["weeks_1_4", "weeks_5_8", "weeks_9_12"]
            ).lower()
            
            # Detect problems in the plan
            plan_has_generic = any(m in plan_text for m in _GENERIC_ACTION_MARKERS)
            plan_has_wrong_role = False
            
            # Check if plan targets a wrong role (e.g., says "QA Engineer" when JD is "Data Scientist")
            if job_title:
                jt_lower = job_title.lower()
                wrong_role_patterns = [
                    "qa engineer", "quality assurance", "frontend developer", "backend developer",
                    "data analyst", "data scientist", "product manager", "devops engineer",
                    "ml engineer", "software engineer", "full stack", "cloud architect",
                    "security engineer", "mobile developer"
                ]
                for role in wrong_role_patterns:
                    if role in plan_text and role not in jt_lower and jt_lower not in role:
                        plan_has_wrong_role = True
                        break
            
            # Check for non-skill terms leaking into the plan (e.g., "Master Bachelor S")
            plan_has_nonsense = any(term in plan_text for term in [
                "bachelor s", "bachelor's", "master s", "to present insights",
                "indeed jobs", "job summary"
            ])
            
            if plan_has_generic or plan_has_wrong_role or plan_has_nonsense:
                # Generate a proper 90-day plan from our skill guidance
                result["action_plan"] = self._generate_quality_action_plan(
                    missing_skills, job_title, domain
                )
        
        return result

    def _generate_quality_action_plan(self, missing_skills: list, job_title: str, domain: str) -> dict:
        """
        Generate a high-quality, domain-specific 90-day action plan
        based on the actual missing skills and detected role.
        """
        # First, try using the domain-specific plan if available
        domain_plan = self._get_domain_specific_action_plan(
            domain, job_title, missing_skills, [], [], [], []
        )
        # If we got a domain-specific plan with real content, use it
        if domain_plan and any(len(domain_plan.get(k, [])) > 0 for k in ["weeks_1_4", "weeks_5_8", "weeks_9_12"]):
            return domain_plan
        
        # Otherwise build from skill guidance
        domain_label = domain.replace('_', ' ').title()
        
        # Filter out non-skill terms from missing skills
        _bad = {"bachelor s", "bachelors", "bachelor", "master s", "masters", "master",
                "phd", "mba", "doctorate", "degree", "to present insights",
                "present insights", "indeed jobs", "job summary", "full time",
                "full-time", "part-time", "remote work", "hybrid"}
        clean_skills = [s for s in missing_skills if s.lower().strip() not in _bad
                        and not re.match(r"^(bachelor|master|phd|doctorate|mba|associate)\b", s.lower())]
        top_skills = clean_skills[:5]
        
        # Get skill-specific guidance for each missing skill
        skill_guides = {}
        for skill in top_skills:
            skill_guides[skill] = self._get_skill_specific_guidance(skill, domain, job_title)
        
        # ── Weeks 1-4: Deep Learning Phase ──
        weeks_1_4 = []
        # Skill-by-skill study plan
        for i, skill in enumerate(top_skills[:3]):
            guide = skill_guides[skill]
            actions = guide["actions"]
            if actions:
                weeks_1_4.append(
                    f"**{skill.title()}** ({guide['timeline']}): {actions[0]}"
                    + (f". Then: {actions[1]}" if len(actions) > 1 else "")
                )
        
        weeks_1_4.append(
            f"Dedicate 10-12 hrs/week to structured learning. "
            f"Focus on building conceptual foundations before hands-on projects."
        )
        weeks_1_4.append(
            f"Set up your development environment for {domain_label}: install all required tools, "
            f"create a GitHub repo for tracking your 90-day progress."
        )
        
        # ── Weeks 5-8: Build & Practice Phase ──
        weeks_5_8 = []
        for skill in top_skills[:3]:
            guide = skill_guides[skill]
            actions = guide["actions"]
            # Pick the project-oriented actions (usually index 2-3)
            project_actions = [a for a in actions[2:4] if a]
            if project_actions:
                weeks_5_8.append(f"**{skill.title()} Practice**: {project_actions[0]}")
        
        weeks_5_8.append(
            f"Build a comprehensive portfolio project combining {', '.join(top_skills[:3])} — "
            f"this should be an end-to-end solution relevant to {job_title} roles."
        )
        weeks_5_8.append(
            f"Document every project with README, architecture diagrams, "
            f"and business context explaining what problem it solves."
        )
        
        # ── Weeks 9-12: Polish & Apply Phase ──
        weeks_9_12 = []
        weeks_9_12.append(
            f"**Resume Enhancement**: Add quantified achievements for each new skill. "
            f"Example format: 'Built [project] using [skills], achieving [measurable result]'."
        )
        
        # Domain-specific interview prep
        interview_prep = {
            "data_science": "Practice ML system design interviews, A/B test case studies, and model evaluation deep-dives.",
            "data_analytics": "Prepare for SQL live coding (30-min complex queries), dashboard case studies, and data storytelling presentations.",
            "product_management": "Master product sense interviews (metrics, trade-offs), estimation questions, and product strategy cases.",
            "frontend": "Practice JavaScript/TypeScript coding challenges, system design for web apps, and component architecture discussions.",
            "backend": "Prepare for system design interviews, API design reviews, and database optimization scenarios.",
            "devops": "Study infrastructure design patterns, incident response scenarios, and CI/CD pipeline architecture questions.",
            "ai_ml": "Practice ML system design, model deployment architecture, and experiment design case studies.",
            "cloud": "Prepare for cloud architecture whiteboard sessions, cost optimization scenarios, and security design reviews.",
            "mobile": "Practice mobile app architecture design, performance optimization scenarios, and platform-specific coding challenges.",
            "sales": "Prepare for behavioral sales interviews: role-play scenarios, pipeline walk-throughs, and 'sell me this pen' exercises. Practice STAR-format answers for quota attainment stories.",
            "hospitality": "Practice hospitality sales interviews: RFP walk-throughs, rate negotiation role-plays, and 90-day business plan presentations. Research each hotel's competitive set.",
            "marketing": "Prepare for marketing case studies: campaign ROI analysis, channel strategy presentations, and metrics-driven decision making exercises.",
            "finance": "Practice financial modeling case studies, Excel proficiency tests, and business valuation walk-throughs.",
            "hr": "Prepare for HR behavioral interviews: conflict resolution scenarios, compliance knowledge, and people strategy case studies.",
        }
        weeks_9_12.append(
            f"**Interview Prep**: {interview_prep.get(domain, f'Practice technical interviews specific to {job_title} roles: system design, coding challenges, and domain-specific case studies.')}"
        )
        
        weeks_9_12.append(
            f"**Portfolio Launch**: Create a professional portfolio site showcasing 3-4 projects with "
            f"live demos, clean code, and business impact documentation."
        )
        weeks_9_12.append(
            f"**Job Applications**: Target {job_title} roles specifically. "
            f"Tailor each application to highlight the skills you've built in weeks 1-8. "
            f"Prepare a 60-second pitch connecting your background to {domain_label} positions."
        )
        
        return {
            "weeks_1_4": weeks_1_4,
            "weeks_5_8": weeks_5_8,
            "weeks_9_12": weeks_9_12
        }

    def _build_local_guidance(self, matched_skills, missing_skills, skill_contexts, error_msg,
                              job_title="Software Developer", domain="general"):
        top_missing = missing_skills[:4]
        strengths = [f"Strong foundation in {skill}" for skill in matched_skills[:5]] or ["Skill analysis completed successfully"]

        priority_skills = []
        all_project_ideas = []
        future_paths_set = set()

        for skill in top_missing:
            ctx = next((item for item in skill_contexts if item.get("skill", "").lower() == skill.lower()), None)
            
            # Get skill-specific guidance (rich, differentiated content per skill)
            specific_guidance = self._get_skill_specific_guidance(skill, domain, job_title)
            
            timeline = ctx.get("estimated_time", specific_guidance["timeline"]) if ctx else specific_guidance["timeline"]
            resources = ctx.get("learning_resources", [])[:4] if ctx else []
            projects = ctx.get("project_ideas", []) if ctx else []
            paths = ctx.get("career_paths", []) if ctx else []
            description = ctx.get("description", "") if ctx else ""
            
            # Use knowledge base resources if available, otherwise use skill-specific generated actions
            actions = resources if resources else specific_guidance["actions"]
            reason = description if description else specific_guidance["reason"]
            
            priority_skills.append({
                "skill": skill,
                "reason": reason,
                "timeline": timeline,
                "actions": actions
            })
            
            for p in projects[:2]:
                all_project_ideas.append((skill, p))
            for path in paths[:2]:
                future_paths_set.add(path)

        # ── Domain-specific project templates ──
        domain_projects = self._get_domain_specific_projects(domain, job_title, top_missing, matched_skills)
        
        recommended_projects = []
        if domain_projects:
            recommended_projects = domain_projects
        elif all_project_ideas:
            for index, (skill, proj_desc) in enumerate(all_project_ideas[:3], start=1):
                recommended_projects.append({
                    "name": f"{skill.title()} — Enterprise Implementation Project",
                    "description": proj_desc,
                    "intuition": f"This project directly addresses the '{skill}' gap identified in the {job_title} job requirements. By building this, you demonstrate practical application of {skill} in a production-like context, which is exactly what hiring managers look for.",
                    "tech_stack": ", ".join([skill] + [s for s in matched_skills[:2] if s.lower() != skill.lower()]),
                    "skills_covered": [skill]
                })
        else:
            recommended_projects = domain_projects or [{
                "name": f"End-to-End {job_title} Portfolio Project",
                "description": f"Build a comprehensive project demonstrating skills needed for a {job_title} role.",
                "intuition": f"This project showcases your ability to deliver a complete solution relevant to {job_title} positions.",
                "tech_stack": ", ".join(top_missing[:3] + matched_skills[:2]),
                "skills_covered": top_missing[:3]
            }]

        # ── Career Paths ──
        immediate_paths = []
        if matched_skills:
            immediate_paths.append(f"Junior/Mid {job_title}")
            if len(matched_skills) > 2:
                immediate_paths.append(f"{matched_skills[0].title()}-focused {domain.replace('_', ' ').title()} roles")
        else:
            immediate_paths = [f"Entry-level {domain.replace('_', ' ').title()} positions"]

        future_paths = list(future_paths_set)[:3]
        if not future_paths:
            domain_label = domain.replace('_', ' ').title()
            if domain in ("sales", "hospitality", "marketing", "finance", "hr", "healthcare", "supply_chain"):
                future_paths = [f"Senior {job_title}", f"Director of {domain_label}"]
            else:
                future_paths = [f"Senior {job_title}", f"Lead/Staff {domain_label} Engineer"]

        # ── Domain-specific 90-Day Action Plan ──
        action_plan = self._get_domain_specific_action_plan(
            domain, job_title, top_missing, matched_skills,
            recommended_projects, future_paths, skill_contexts
        )

        return {
            "career_summary": (
                f"Based on your profile analysis for the {job_title} role, you match many core requirements. "
                f"Focus on mastering {', '.join(top_missing[:2])} over the next 90 days to significantly boost your candidacy. "
                f"Your existing strengths in {', '.join(matched_skills[:3])} provide a solid foundation."
            ),
            "error": self._format_error_message(error_msg),
            "fallback_mode": True,
            "strengths": strengths,
            "priority_skills": priority_skills,
            "recommended_projects": recommended_projects,
            "career_paths": {"immediate": immediate_paths, "after_upskilling": future_paths},
            "action_plan": action_plan
        }

    def _get_skill_specific_guidance(self, skill: str, domain: str, job_title: str) -> dict:
        """
        Generate rich, differentiated guidance for a specific skill.
        Returns a dict with 'reason', 'timeline', and 'actions' that are unique
        to the skill — NEVER generic boilerplate.
        """
        skill_lower = skill.lower().strip()
        domain_label = domain.replace('_', ' ')
        
        # ── Comprehensive skill guidance database ──
        _SKILL_GUIDANCE = {
            # ─── Data & Analytics Skills ───
            "interactive dashboards": {
                "reason": f"Interactive dashboards are the primary interface between data teams and business stakeholders. For a {job_title} role, building dashboards that allow drill-downs, filters, and real-time KPI tracking is essential — it transforms raw data into executive-level decision support.",
                "timeline": "4-6 weeks with dedicated daily practice",
                "actions": [
                    "Master Tableau Desktop: complete Tableau's free e-learning on calculated fields, LOD expressions, and dashboard actions",
                    "Build 3 interactive dashboards using Kaggle datasets (sales analytics, HR metrics, supply chain)",
                    "Learn Power BI DAX formulas: CALCULATE, SUMX, time intelligence functions, and create parameterized reports with slicers",
                    "Study 'Storytelling with Data' by Cole Nussbaumer Knaflic for dashboard UX best practices",
                    "Publish your best dashboards to Tableau Public and link them in your portfolio"
                ]
            },
            "tableau": {
                "reason": f"Tableau is the industry-leading visual analytics platform used by 86% of Fortune 500 companies. For a {job_title}, proficiency in Tableau means you can independently create executive dashboards, blend data sources, and deliver actionable visual insights without relying on engineering teams.",
                "timeline": "3-5 weeks for intermediate proficiency",
                "actions": [
                    "Complete Tableau Desktop Specialist certification prep (free on Tableau's e-learning platform)",
                    "Master core concepts: calculated fields, table calculations, LOD expressions (FIXED, INCLUDE, EXCLUDE)",
                    "Build dashboards with real datasets: connect to CSV, Excel, and SQL databases; practice data blending",
                    "Learn dashboard design patterns: KPI scorecards, trend analysis, geo-mapping, and drill-down hierarchies",
                    "Complete 5 Makeover Monday challenges and publish results to Tableau Public"
                ]
            },
            "power bi": {
                "reason": f"Power BI is Microsoft's flagship analytics tool deeply integrated with the Office 365 ecosystem. For a {job_title}, Power BI skills let you build self-service analytics, automate reports via Power Automate, and share insights across the organization through Teams and SharePoint.",
                "timeline": "3-5 weeks for working proficiency",
                "actions": [
                    "Complete Microsoft's free Power BI learning path on Microsoft Learn (PL-300 prep)",
                    "Master DAX fundamentals: CALCULATE, FILTER, ALL, DATEADD, and time intelligence patterns",
                    "Build data models with Star Schema design: fact tables, dimension tables, and relationships",
                    "Create 3 reports with slicers, bookmarks, drill-through pages, and Row-Level Security (RLS)",
                    "Practice Power Query M for ETL: merge queries, pivot/unpivot, and custom columns"
                ]
            },
            "predictive models": {
                "reason": f"Predictive modeling enables data-driven forecasting — from customer churn to demand planning. For a {job_title}, understanding how to build, validate, and deploy predictive models means you can drive proactive decision-making rather than reactive reporting.",
                "timeline": "6-10 weeks for hands-on proficiency",
                "actions": [
                    "Study supervised learning fundamentals: linear/logistic regression, decision trees, random forests, XGBoost",
                    "Complete Andrew Ng's Machine Learning Specialization on Coursera (focus on Weeks 1-6)",
                    "Build 3 end-to-end prediction projects in Python: churn prediction, price forecasting, and classification",
                    "Master model evaluation: cross-validation, ROC-AUC, precision-recall trade-offs, confusion matrices",
                    "Learn scikit-learn pipelines, feature engineering (encoding, scaling, imputation), and hyperparameter tuning with GridSearchCV"
                ]
            },
            "hypothesis testing": {
                "reason": f"Hypothesis testing is the statistical backbone of data-driven decisions. For a {job_title}, it enables you to rigorously validate A/B test results, measure feature impact, and make evidence-based recommendations with quantified confidence levels.",
                "timeline": "3-4 weeks for applied proficiency",
                "actions": [
                    "Study core statistical tests: t-tests (paired/independent), chi-square, ANOVA, and Mann-Whitney U",
                    "Complete Khan Academy Statistics & Probability course (focus on inference sections)",
                    "Practice with Python's scipy.stats: run hypothesis tests on real datasets from Kaggle",
                    "Master A/B testing methodology: sample size calculation, statistical power, p-values, and confidence intervals",
                    "Build a Python notebook demonstrating 5 different hypothesis tests with real business scenarios"
                ]
            },
            "mathematics": {
                "reason": f"Strong mathematical foundations underpin all quantitative work — from statistical modeling to algorithm design. For a {job_title}, applied mathematics enables you to understand model assumptions, optimize business metrics, and communicate findings with statistical rigor.",
                "timeline": "6-10 weeks for applied foundations",
                "actions": [
                    "Study linear algebra essentials: vectors, matrices, eigenvalues (3Blue1Brown's Essence of Linear Algebra)",
                    "Master probability & statistics: Bayes' theorem, distributions, Central Limit Theorem, hypothesis testing",
                    "Complete Khan Academy's Multivariable Calculus and Statistics courses",
                    "Apply math concepts in Python: implement gradient descent, matrix operations with NumPy",
                    "Read 'Mathematics for Machine Learning' (free PDF by Deisenroth, Faisal, Ong) — focus on Chapters 2, 5, 6"
                ]
            },
            # ─── IoT & Hardware Skills ───
            "iot devices": {
                "reason": f"IoT connects physical devices to cloud analytics, enabling real-time monitoring, predictive maintenance, and smart automation. For a {job_title}, IoT expertise means you can architect end-to-end solutions from sensor data collection through edge processing to cloud dashboards.",
                "timeline": "6-8 weeks for prototyping proficiency",
                "actions": [
                    "Start with Arduino/Raspberry Pi: build 3 sensor projects (temperature, motion, GPS tracking)",
                    "Learn MQTT protocol and set up a Mosquitto broker for device-to-cloud communication",
                    "Study AWS IoT Core or Azure IoT Hub: device provisioning, message routing, and digital twins",
                    "Build an end-to-end IoT dashboard: sensors → MQTT → cloud → real-time visualization with Grafana",
                    "Complete AWS IoT Specialization or HiveMQ's free MQTT course"
                ]
            },
            "iot": {
                "reason": f"IoT connects physical devices to cloud analytics, enabling real-time monitoring, predictive maintenance, and smart automation. For a {job_title}, IoT expertise means you can architect end-to-end solutions from sensor data collection through edge processing to cloud dashboards.",
                "timeline": "6-8 weeks for prototyping proficiency",
                "actions": [
                    "Start with Arduino/Raspberry Pi: build 3 sensor projects (temperature, motion, GPS tracking)",
                    "Learn MQTT protocol and set up a Mosquitto broker for device-to-cloud communication",
                    "Study AWS IoT Core or Azure IoT Hub: device provisioning, message routing, and digital twins",
                    "Build an end-to-end IoT dashboard: sensors → MQTT → cloud → real-time visualization with Grafana",
                    "Complete AWS IoT Specialization or HiveMQ's free MQTT course"
                ]
            },
            # ─── Product Management Skills ───
            "product lifecycle": {
                "reason": f"Product lifecycle management spans ideation → launch → growth → sunset. For a {job_title}, understanding each phase means you can prioritize features with frameworks like RICE/ICE, manage technical debt strategically, and drive revenue at every stage.",
                "timeline": "4-6 weeks of study + real-world application",
                "actions": [
                    "Read 'Inspired' by Marty Cagan — focus on product discovery and delivery chapters",
                    "Study prioritization frameworks: RICE, ICE, MoSCoW, Kano Model — practice applying them to real products",
                    "Learn product analytics: set up funnel analysis, cohort tracking, and retention metrics in Amplitude or Mixpanel",
                    "Create a complete product lifecycle document for a real product: vision → roadmap → launch metrics → iteration plan",
                    "Complete Reforge's Product Strategy course or Product School's free PM certification"
                ]
            },
            "a/b testing": {
                "reason": f"A/B testing is how data-driven companies validate product decisions with statistical confidence. For a {job_title}, designing and interpreting experiments means you can quantify feature impact, reduce risk of bad launches, and make evidence-based prioritization decisions.",
                "timeline": "3-4 weeks for practical expertise",
                "actions": [
                    "Master experimental design: control vs. treatment groups, randomization, sample size calculators (Evan Miller's tool)",
                    "Study statistical foundations: p-values, confidence intervals, statistical power, Type I/II errors",
                    "Learn platform-specific tools: Google Optimize, Optimizely, or LaunchDarkly for feature flags",
                    "Build a Python A/B test analysis pipeline: data collection → significance testing → effect size estimation",
                    "Complete Udacity's A/B Testing by Google course (free) and analyze 3 real experiment datasets"
                ]
            },
            "product roadmap": {
                "reason": f"A product roadmap translates business strategy into a sequenced execution plan. For a {job_title}, owning the roadmap means aligning engineering, design, and business stakeholders on priorities, timelines, and trade-offs while maintaining strategic flexibility.",
                "timeline": "3-5 weeks to learn frameworks + practice",
                "actions": [
                    "Study roadmap types: timeline-based, Now-Next-Later, theme-based, and outcome-driven roadmaps",
                    "Learn tools: Jira Advanced Roadmaps, Productboard, Aha!, or Notion for roadmap visualization",
                    "Practice prioritization: apply RICE scoring to 20+ feature requests from a real product backlog",
                    "Create 3 roadmaps for real products at different maturity stages (startup, growth, enterprise)",
                    "Read 'Product Roadmaps Relaunched' by Lombardo et al. and study how Spotify/Airbnb structure their roadmaps"
                ]
            },
            "demand forecasting": {
                "reason": f"Demand forecasting enables organizations to optimize inventory, staffing, and resource allocation. For a {job_title}, accurate forecasting directly impacts revenue, reduces waste, and informs strategic planning across supply chain, finance, and operations.",
                "timeline": "5-8 weeks for applied proficiency",
                "actions": [
                    "Study time series fundamentals: trend, seasonality, stationarity, autocorrelation (ACF/PACF)",
                    "Master forecasting methods: ARIMA, SARIMA, Exponential Smoothing (Holt-Winters), and Prophet",
                    "Complete a demand forecasting project in Python using Facebook Prophet and real retail/sales data from Kaggle",
                    "Learn evaluation metrics: MAPE, RMSE, MAE, and backtesting with rolling cross-validation",
                    "Study 'Forecasting: Principles and Practice' (free online textbook by Hyndman & Athanasopoulos)"
                ]
            },
            # ─── DevOps & Infrastructure ───
            "docker": {
                "reason": f"Docker enables consistent, reproducible environments across development, testing, and production. For a {job_title}, containerization skills mean you can package applications reliably, reduce 'works on my machine' issues, and deploy microservices at scale.",
                "timeline": "2-3 weeks for working proficiency",
                "actions": [
                    "Complete Docker's official Getting Started tutorial and understand images, containers, volumes, and networks",
                    "Write Dockerfiles for 3 different application stacks (Python/Flask, Node/Express, Java/Spring)",
                    "Master Docker Compose: multi-container setups with databases, caches, and application servers",
                    "Learn Docker best practices: multi-stage builds, layer caching, security scanning with Trivy",
                    "Deploy a dockerized application to AWS ECS or Azure Container Instances"
                ]
            },
            "kubernetes": {
                "reason": f"Kubernetes orchestrates containerized workloads at scale — handling deployment, scaling, and self-healing. For a {job_title}, K8s expertise is essential for managing production microservices, implementing blue-green deployments, and ensuring high availability.",
                "timeline": "6-10 weeks for intermediate proficiency",
                "actions": [
                    "Complete Kubernetes official tutorials (kubernetes.io) and set up a local cluster with minikube or kind",
                    "Master core objects: Pods, Deployments, Services, ConfigMaps, Secrets, Ingress, and PersistentVolumes",
                    "Deploy a multi-service application with Helm charts and practice rolling updates",
                    "Study monitoring and observability: Prometheus, Grafana dashboards, and kubectl debugging",
                    "Prepare for CKA certification: complete KodeKloud's CKA course with hands-on labs"
                ]
            },
            "ci/cd": {
                "reason": f"CI/CD automates the build-test-deploy pipeline, enabling rapid and reliable software delivery. For a {job_title}, implementing CI/CD means faster release cycles, fewer production bugs, and the ability to ship features confidently multiple times per day.",
                "timeline": "3-4 weeks for practical setup experience",
                "actions": [
                    "Set up a complete CI/CD pipeline with GitHub Actions: build → lint → test → deploy on every push",
                    "Learn pipeline stages: unit testing, integration testing, Docker image building, and deployment triggers",
                    "Study Jenkins or GitLab CI as enterprise alternatives: understand pipeline-as-code (Jenkinsfile, .gitlab-ci.yml)",
                    "Implement environment promotion: dev → staging → production with manual approval gates",
                    "Add code quality checks: SonarQube analysis, test coverage thresholds, and dependency vulnerability scanning"
                ]
            },
            # ─── Frontend Skills ───
            "react": {
                "reason": f"React dominates the frontend ecosystem with 40%+ market share and is used by Meta, Netflix, Airbnb, and thousands of companies. For a {job_title}, React proficiency means you can build complex, performant UIs with component-based architecture and a massive ecosystem.",
                "timeline": "4-6 weeks for job-ready proficiency",
                "actions": [
                    "Complete the official React docs (react.dev) — especially the new hooks-first tutorial",
                    "Master core hooks: useState, useEffect, useContext, useReducer, useMemo, useCallback, useRef",
                    "Build 3 projects: a task manager (CRUD), a weather app (API calls), a dashboard (state management with Zustand/Redux)",
                    "Learn React Router v6, React Query/TanStack Query for data fetching, and form handling with React Hook Form",
                    "Study testing: Jest + React Testing Library for unit/integration tests, Cypress for E2E tests"
                ]
            },
            "typescript": {
                "reason": f"TypeScript adds static type safety to JavaScript, catching bugs at compile time and enabling better IDE support. For a {job_title}, TypeScript is now the industry standard for production codebases — it's required by most frontend and full-stack roles.",
                "timeline": "2-4 weeks to become productive",
                "actions": [
                    "Complete TypeScript's official handbook (typescriptlang.org/docs/handbook)",
                    "Master type system fundamentals: interfaces, generics, union/intersection types, type guards, utility types",
                    "Convert an existing JavaScript project to TypeScript to practice migration patterns",
                    "Study advanced patterns: discriminated unions, mapped types, conditional types, and template literal types",
                    "Complete Matt Pocock's Total TypeScript free tutorials (totaltypescript.com)"
                ]
            },
            "javascript": {
                "reason": f"JavaScript is the universal language of the web, running in every browser and powering server-side with Node.js. For a {job_title}, deep JS knowledge — closures, async patterns, event loop — separates senior engineers from juniors.",
                "timeline": "3-6 weeks for advanced proficiency",
                "actions": [
                    "Study 'You Don't Know JS Yet' by Kyle Simpson (free on GitHub) — focus on Scope, Closures, and Async",
                    "Master ES6+ features: destructuring, spread/rest, modules, Promises, async/await, generators",
                    "Practice on LeetCode/HackerRank: solve 50 problems in JavaScript focusing on arrays, strings, and recursion",
                    "Build a vanilla JS project without frameworks to solidify DOM manipulation and event handling",
                    "Learn performance optimization: Web Workers, requestAnimationFrame, and memory leak detection with Chrome DevTools"
                ]
            },
            # ─── Backend & Data Engineering ───
            "sql": {
                "reason": f"SQL is the lingua franca of data — every database, data warehouse, and analytics tool relies on it. For a {job_title}, advanced SQL means you can independently query complex datasets, optimize performance, and build reporting pipelines without engineering bottlenecks.",
                "timeline": "2-4 weeks for advanced query proficiency",
                "actions": [
                    "Master advanced SQL: window functions (ROW_NUMBER, RANK, LAG/LEAD), CTEs, subqueries, and CASE statements",
                    "Complete Mode Analytics SQL tutorial (free) and practice on LeetCode SQL problems (50+ medium/hard)",
                    "Learn query optimization: EXPLAIN plans, indexing strategies, and avoiding N+1 query patterns",
                    "Study database-specific features: PostgreSQL (JSONB, arrays), MySQL (full-text search), or BigQuery (partitioning)",
                    "Build 3 analytical queries: cohort analysis, funnel metrics, and rolling averages on real datasets"
                ]
            },
            "python": {
                "reason": f"Python is the dominant language for data science, ML, automation, and backend development. For a {job_title}, Python fluency enables rapid prototyping, data manipulation with pandas, ML model building, and API development — making you versatile across teams.",
                "timeline": "3-6 weeks for job-ready proficiency",
                "actions": [
                    "Master core libraries: pandas (data wrangling), NumPy (numerical computing), matplotlib/seaborn (visualization)",
                    "Complete 'Python for Data Analysis' by Wes McKinney — focus on chapters on pandas and data cleaning",
                    "Build 3 automation scripts: web scraping (BeautifulSoup/Scrapy), email automation, and data pipeline ETL",
                    "Learn FastAPI or Flask for building REST APIs with proper error handling and authentication",
                    "Practice on HackerRank Python challenges and contribute to an open-source Python project on GitHub"
                ]
            },
            # ─── Cloud & Architecture ───
            "aws": {
                "reason": f"AWS commands 32% of the cloud market and is the default infrastructure for startups to enterprises. For a {job_title}, AWS skills mean you can architect scalable, cost-efficient solutions and speak the same language as the infrastructure teams you'll collaborate with.",
                "timeline": "6-8 weeks for practitioner-level proficiency",
                "actions": [
                    "Complete AWS Cloud Practitioner certification prep (free AWS Skill Builder courses)",
                    "Master core services: EC2, S3, RDS, Lambda, API Gateway, IAM, VPC, CloudWatch",
                    "Build a serverless application: API Gateway → Lambda → DynamoDB with proper IAM roles",
                    "Study the AWS Well-Architected Framework: reliability, security, cost optimization pillars",
                    "Complete Stephane Maarek's AWS Solutions Architect course on Udemy and practice with free tier"
                ]
            },
            "azure": {
                "reason": f"Azure is the fastest-growing cloud platform, deeply integrated with Microsoft's enterprise ecosystem. For a {job_title}, Azure proficiency is critical in organizations using Office 365, Active Directory, and Microsoft's data stack (Synapse, Power BI, Azure ML).",
                "timeline": "5-8 weeks for AZ-900/DP-900 level",
                "actions": [
                    "Complete Microsoft Learn's AZ-900 (Azure Fundamentals) learning path — free with hands-on labs",
                    "Master core services: App Service, Azure Functions, Blob Storage, SQL Database, Azure AD",
                    "Build a full-stack app deployed on Azure: App Service + SQL Database + Azure AD authentication",
                    "Study Azure DevOps: CI/CD pipelines, Boards for work tracking, and Artifacts for package management",
                    "Get AZ-900 certified (free vouchers often available through Microsoft Virtual Training Days)"
                ]
            },
            # ─── Agile & Project Management ───
            "jira": {
                "reason": f"Jira is the standard project management tool for Agile teams worldwide. For a {job_title}, Jira proficiency means you can manage sprints, track epics and user stories, configure workflows, and generate velocity reports that keep teams aligned and stakeholders informed.",
                "timeline": "1-2 weeks for proficient daily use",
                "actions": [
                    "Complete Atlassian's free Jira Fundamentals course on Atlassian University",
                    "Master Jira concepts: Epics → Stories → Tasks → Subtasks hierarchy, sprint planning, and backlog grooming",
                    "Learn JQL (Jira Query Language) for advanced issue searching and creating custom filters/dashboards",
                    "Practice board configuration: create Scrum and Kanban boards with custom workflows and swim lanes",
                    "Set up automation rules: auto-assign, transition triggers, and Slack/Teams notifications"
                ]
            },
            "agile": {
                "reason": f"Agile methodology is the dominant software development approach, used by 71% of organizations. For a {job_title}, understanding Scrum ceremonies, sprint planning, and backlog management is non-negotiable — it's how cross-functional teams deliver value iteratively.",
                "timeline": "2-3 weeks for framework knowledge + ongoing practice",
                "actions": [
                    "Study the Scrum Guide (scrumguides.org) — understand roles (PO, SM, Dev Team), events, and artifacts",
                    "Learn Kanban principles: WIP limits, flow metrics (lead time, cycle time, throughput), and board design",
                    "Complete Google's free Agile Project Management course on Coursera (part of PM Certificate)",
                    "Practice sprint planning: write user stories (As a..., I want..., So that...) with acceptance criteria",
                    "Prepare for PSM-I or CSM certification to validate your Agile knowledge"
                ]
            },
            # ─── Data Visualization & Reporting ───
            "data visualization": {
                "reason": f"Data visualization converts complex datasets into intuitive visual stories. For a {job_title}, strong visualization skills mean you can communicate insights to non-technical stakeholders, influence decisions, and make data accessible across the organization.",
                "timeline": "3-5 weeks for solid proficiency",
                "actions": [
                    "Study 'Storytelling with Data' by Cole Nussbaumer Knaflic — master the 6 visualization principles",
                    "Learn Python visualization: matplotlib, seaborn (statistical), plotly (interactive), and Altair (declarative)",
                    "Master chart selection: when to use bar vs. line vs. scatter vs. heatmap vs. treemap for different data types",
                    "Build an interactive visualization portfolio using Plotly Dash or Streamlit with 3 real-world datasets",
                    "Complete Tableau Desktop Specialist prep or take DataCamp's Data Visualization track"
                ]
            },
            "model building optimization design": {
                "reason": f"Model optimization is the art of improving ML model accuracy, speed, and generalization. For a {job_title}, understanding hyperparameter tuning, feature selection, and model architecture design separates basic practitioners from engineers who can deliver production-grade ML systems.",
                "timeline": "6-10 weeks for applied optimization skills",
                "actions": [
                    "Study feature engineering techniques: encoding strategies, feature selection (mutual information, L1), and feature stores",
                    "Master hyperparameter optimization: GridSearchCV, RandomSearchCV, Bayesian optimization with Optuna",
                    "Learn model evaluation beyond accuracy: cross-validation strategies, learning curves, bias-variance analysis",
                    "Practice model compression: pruning, quantization, knowledge distillation for production deployment",
                    "Complete Kaggle's Intermediate Machine Learning course and enter 2 competitions focusing on model optimization"
                ]
            },
            # ─── Collaboration & Soft Skills (Technical) ───
            "git": {
                "reason": f"Git is the universal version control system powering all modern software collaboration. For a {job_title}, mastering Git means you can manage feature branches, resolve merge conflicts, conduct code reviews, and maintain clean project history — essential for any team environment.",
                "timeline": "1-2 weeks for daily workflow proficiency",
                "actions": [
                    "Complete Atlassian's Git tutorial series (atlassian.com/git) — focus on branching strategies",
                    "Master essential workflows: feature branching, Git Flow, trunk-based development, and rebasing vs. merging",
                    "Practice conflict resolution: create intentional merge conflicts and resolve them using VS Code's merge editor",
                    "Learn advanced Git: interactive rebase, cherry-pick, bisect for debugging, stash, and reflog for recovery",
                    "Set up Git hooks and learn conventional commits for clean project history"
                ]
            },
            "graphql": {
                "reason": f"GraphQL provides a flexible query language that lets clients request exactly the data they need, eliminating over-fetching. For a {job_title}, GraphQL skills enable you to design efficient APIs for complex frontends, mobile apps, and microservice architectures.",
                "timeline": "3-4 weeks for working proficiency",
                "actions": [
                    "Complete the official GraphQL tutorials at graphql.org — understand schemas, types, queries, mutations, and subscriptions",
                    "Build a GraphQL API using Apollo Server (Node.js) or Strawberry (Python) with a PostgreSQL database",
                    "Master client-side GraphQL: Apollo Client for React with caching, optimistic updates, and pagination",
                    "Study schema design best practices: pagination (cursor vs. offset), error handling, and N+1 query prevention with DataLoader",
                    "Implement a full-stack app: GraphQL API → React frontend with authentication and real-time subscriptions"
                ]
            },
            "rest api": {
                "reason": f"REST APIs are the backbone of modern software architecture, enabling communication between frontend, backend, mobile, and third-party services. For a {job_title}, designing and consuming RESTful APIs is a fundamental daily skill.",
                "timeline": "2-3 weeks for design and implementation proficiency",
                "actions": [
                    "Study REST principles: resources, HTTP methods (GET/POST/PUT/PATCH/DELETE), status codes, and HATEOAS",
                    "Build 3 REST APIs using different frameworks: FastAPI (Python), Express (Node.js), or Spring Boot (Java)",
                    "Master API authentication: JWT tokens, OAuth 2.0 flows, and API key management",
                    "Learn API documentation with OpenAPI/Swagger and test APIs with Postman collections",
                    "Study API design best practices: versioning, pagination, filtering, rate limiting, and error handling patterns"
                ]
            },
            
            # ─── Sales / Hospitality / Business Skills ───
            "cold calling": {
                "reason": f"Cold calling is the foundation of outbound sales — the ability to initiate conversations with prospects who haven't expressed prior interest. For a {job_title}, mastering cold calling techniques directly drives new account acquisition and revenue pipeline.",
                "timeline": "2-3 weeks for technique mastery, ongoing practice",
                "actions": [
                    "Study the SPIN Selling framework (Situation, Problem, Implication, Need-Payoff) by Neil Rackham",
                    "Script and practice 5 different opening hooks tailored to corporate travel buyers, gatekeepers, and decision-makers",
                    "Role-play cold call scenarios: handling objections ('we already have a vendor', 'not interested', 'send me an email')",
                    "Learn to research prospects before calling: LinkedIn, company travel policies, recent RFPs, and industry news",
                    "Track call metrics (dials, connects, meetings booked) daily and refine your approach based on conversion rates"
                ]
            },
            "crm systems": {
                "reason": f"CRM (Customer Relationship Management) systems are the operational backbone of any sales organization. For a {job_title}, CRM proficiency means you can track deals, manage pipelines, forecast revenue, and demonstrate ROI to management — it's non-negotiable in modern B2B sales.",
                "timeline": "3-4 weeks for working proficiency",
                "actions": [
                    "Complete Salesforce Trailhead 'Sales Cloud for Sales Reps' module (free, ~20 hours)",
                    "Alternatively, master HubSpot CRM free certification if your target companies use HubSpot",
                    "Practice core CRM workflows: lead entry, opportunity creation, pipeline staging, activity logging, and report generation",
                    "Learn to build custom reports and dashboards: pipeline by stage, win rate by source, revenue forecast by quarter",
                    "Study CRM best practices: data hygiene, lead scoring, automated follow-up sequences, and integration with email/calendar"
                ]
            },
            "negotiation": {
                "reason": f"Negotiation is the skill that directly impacts deal size, contract terms, and profit margins. For a {job_title}, strong negotiation skills mean securing better rates, longer contract terms, and more favorable conditions — directly translating to higher revenue and commission.",
                "timeline": "4-6 weeks of study + ongoing real-world practice",
                "actions": [
                    "Read 'Never Split the Difference' by Chris Voss — master tactical empathy, labeling, and calibrated questions",
                    "Study BATNA (Best Alternative to Negotiated Agreement) framework and practice calculating walk-away points for rate negotiations",
                    "Role-play 5 hotel rate negotiation scenarios: initial proposal, counter-offer, volume discount, multi-year deal, last-minute concession",
                    "Learn to create value beyond price: room upgrades, meeting space, F&B credits, late check-out as negotiation levers",
                    "Practice building and presenting business cases that justify your proposed rates with occupancy data and competitive comparisons"
                ]
            },
            "ms office": {
                "reason": f"MS Office proficiency (especially Excel, PowerPoint, and Outlook) is the baseline productivity expectation in B2B sales. For a {job_title}, advanced Excel skills enable revenue analysis and forecasting, while PowerPoint creates compelling client presentations that close deals.",
                "timeline": "2-3 weeks for advanced proficiency",
                "actions": [
                    "Master Excel for sales: VLOOKUP/XLOOKUP, pivot tables, conditional formatting, and creating revenue tracking templates",
                    "Learn Excel financial functions: NPV, IRR, scenario analysis for rate proposals and ROI presentations",
                    "Build 3 professional PowerPoint sales decks: hotel overview, corporate rate proposal, and quarterly business review (QBR)",
                    "Master Outlook: calendar management, email templates, distribution lists, and automated follow-up rules",
                    "Complete Microsoft Office Specialist (MOS) Excel certification prep for credibility on your resume"
                ]
            },
            "hotel sales": {
                "reason": f"Hotel sales expertise combines hospitality knowledge with B2B selling skills — understanding room inventory, rate management, contract structures, and the competitive landscape. For a {job_title}, this domain knowledge differentiates you from generic sales candidates.",
                "timeline": "4-6 weeks of intensive study + industry immersion",
                "actions": [
                    "Study hotel revenue management fundamentals: ADR (Average Daily Rate), RevPAR, occupancy rates, and yield management principles",
                    "Learn hotel distribution channels: GDS (Amadeus, Sabre), OTAs (Booking.com, Expedia), direct booking, and TMC partnerships",
                    "Master corporate rate negotiation: RFP process, last room availability (LRA), dynamic pricing vs. static rates, and volume commitments",
                    "Study STR (Smith Travel Research) reports to understand comp set analysis and market positioning",
                    "Network through HSMAI (Hospitality Sales & Marketing Association International) — attend local chapter events and earn HSMAI certification"
                ]
            },
            "pipedrive": {
                "reason": f"Pipedrive is a sales-focused CRM designed for pipeline management and deal tracking. For a {job_title}, Pipedrive proficiency means you can manage your entire sales funnel visually, automate follow-ups, and generate accurate revenue forecasts.",
                "timeline": "1-2 weeks for full proficiency",
                "actions": [
                    "Complete Pipedrive Academy courses (free): Pipeline Management, Activity-Based Selling, and Reporting",
                    "Set up a complete sales pipeline: define stages from Lead → Qualified → Proposal → Negotiation → Won/Lost",
                    "Master Pipedrive automations: automated email sequences, activity reminders, and deal stage triggers",
                    "Learn to build custom dashboards: deals by stage, revenue forecast, activity metrics, and win/loss analysis",
                    "Practice integrating Pipedrive with email, calendar, and other tools via Zapier or native integrations"
                ]
            },
            "amadeus": {
                "reason": f"Amadeus is one of the world's largest GDS (Global Distribution System) platforms, connecting hotels with travel agencies and corporate travel departments worldwide. For a {job_title}, Amadeus proficiency opens your hotel to a massive distribution channel.",
                "timeline": "2-3 weeks for operational proficiency",
                "actions": [
                    "Complete Amadeus e-Learning Academy: Hotel Sales and GDS connectivity modules",
                    "Learn Amadeus hotel product: rate loading, availability management, and corporate rate codes (BT/IT rates)",
                    "Practice analyzing Amadeus booking data: identify top-producing travel agencies, corporate accounts, and booking patterns",
                    "Understand Amadeus vs. Sabre vs. Travelport: competitive landscape of GDS platforms and their market share",
                    "Study how TMCs use Amadeus for corporate travel booking and how to optimize your hotel's GDS presence"
                ]
            },
            "concur travel": {
                "reason": f"SAP Concur is the world's leading corporate travel and expense management platform, used by 48,000+ companies. For a {job_title}, understanding Concur means you can interface with corporate travel buyers and understand their booking workflows.",
                "timeline": "1-2 weeks for conceptual understanding",
                "actions": [
                    "Study SAP Concur Travel & Expense platform: how corporate travelers book, approve, and expense travel",
                    "Understand Concur hotel program participation: preferred rates, content sourcing, and e-folio integration",
                    "Learn how corporate travel policies in Concur affect hotel selection: rate caps, preferred vendor flags, and duty-of-care requirements",
                    "Practice analyzing Concur travel data patterns to identify upsell and cross-sell opportunities for your property",
                    "Research how to get your hotel listed as a preferred property in Concur corporate programs"
                ]
            },
            "marketo": {
                "reason": f"Marketo (Adobe Marketo Engage) is an enterprise marketing automation platform used for lead nurturing, email campaigns, and attribution. For a {job_title}, Marketo skills enable sophisticated lead scoring and campaign management that directly feed sales pipeline.",
                "timeline": "3-4 weeks for working proficiency",
                "actions": [
                    "Complete Marketo Fundamentals on Adobe Experience League (free)",
                    "Master core features: Smart Lists, Smart Campaigns, lead scoring models, and email program setup",
                    "Build 3 automated campaigns: welcome series, lead nurture sequence, and re-engagement campaign with A/B testing",
                    "Learn Marketo-CRM integration: lead sync, attribution reporting, and marketing-sales alignment dashboards",
                    "Study Marketo Certified Expert (MCE) exam prep materials for resume credibility"
                ]
            },
            "salesforce": {
                "reason": f"Salesforce is the world's #1 CRM platform with 23% market share. For a {job_title}, Salesforce proficiency is often a hard requirement — it's the system of record for pipeline management, account tracking, forecasting, and cross-team collaboration.",
                "timeline": "4-6 weeks for Sales Cloud proficiency",
                "actions": [
                    "Complete Salesforce Trailhead: 'Sales Cloud for Sales Reps' and 'Reports & Dashboards' trails (free, ~40 hours)",
                    "Master core sales workflows: Lead → Opportunity → Quote → Close, with proper stage management and activity logging",
                    "Build custom reports and dashboards: pipeline by stage, win rate by source, revenue forecast, and activity metrics",
                    "Learn Salesforce automation: workflow rules, process builder, email templates, and approval processes",
                    "Earn Salesforce Administrator certification (strongly boosts resume for any sales role)"
                ]
            },
        }
        
        # Check for direct match
        if skill_lower in _SKILL_GUIDANCE:
            return _SKILL_GUIDANCE[skill_lower]
        
        # Check for partial/fuzzy match (e.g., "tableau, power bi" → "tableau")
        for known_skill, guidance in _SKILL_GUIDANCE.items():
            if known_skill in skill_lower or skill_lower in known_skill:
                return guidance
        
        # Check for keyword-based matching in multi-word skills
        skill_words = set(skill_lower.split())
        best_match = None
        best_overlap = 0
        for known_skill, guidance in _SKILL_GUIDANCE.items():
            known_words = set(known_skill.split())
            overlap = len(skill_words & known_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = guidance
        if best_match and best_overlap >= 1:
            return best_match
        
        # ── Domain-aware intelligent fallback ──
        # Generate meaningful guidance based on the skill name + domain context
        domain_context = {
            "frontend": {
                "focus": "UI components, responsive design, accessibility, and state management",
                "tools": "React DevTools, Chrome DevTools, Lighthouse, Storybook",
                "resources": "MDN Web Docs, Frontend Masters, Kent C. Dodds' Epic React",
            },
            "backend": {
                "focus": "API design, database optimization, caching, and microservice patterns",
                "tools": "Postman, Docker, Redis, PostgreSQL, monitoring with Datadog",
                "resources": "System Design Primer (GitHub), Designing Data-Intensive Applications by Kleppmann",
            },
            "data_science": {
                "focus": "statistical modeling, feature engineering, and model deployment",
                "tools": "Jupyter, scikit-learn, TensorFlow/PyTorch, MLflow",
                "resources": "Kaggle Learn, fast.ai, Andrew Ng's ML Specialization",
            },
            "data_analytics": {
                "focus": "data querying, visualization, statistical analysis, and business intelligence",
                "tools": "SQL, Python (pandas), Tableau/Power BI, Excel advanced features",
                "resources": "Mode Analytics SQL Tutorial, DataCamp, Google Data Analytics Certificate",
            },
            "product_management": {
                "focus": "product strategy, user research, metrics-driven decision making, and roadmap planning",
                "tools": "Jira, Amplitude/Mixpanel, Figma, Miro, Productboard",
                "resources": "Reforge courses, 'Inspired' by Marty Cagan, Product School resources",
            },
            "devops": {
                "focus": "infrastructure automation, container orchestration, CI/CD, and monitoring",
                "tools": "Terraform, Ansible, Kubernetes, Prometheus, GitHub Actions",
                "resources": "'The Phoenix Project' by Gene Kim, KodeKloud, Linux Academy",
            },
            "ai_ml": {
                "focus": "model training, MLOps, experiment tracking, and production deployment",
                "tools": "MLflow, Weights & Biases, Kubeflow, TensorFlow Serving",
                "resources": "deeplearning.ai, fast.ai, Hugging Face courses",
            },
            "sales": {
                "focus": "pipeline management, prospect qualification, negotiation, and closing techniques",
                "tools": "Salesforce/HubSpot/Pipedrive CRM, LinkedIn Sales Navigator, Outreach.io, Gong.io",
                "resources": "'SPIN Selling' by Neil Rackham, 'The Challenger Sale', Sandler Sales Training",
            },
            "hospitality": {
                "focus": "hotel revenue management, corporate rate negotiation, GDS distribution, and client relationship management",
                "tools": "Amadeus, Sabre, Opera PMS, Concur Travel, STR reports, Delphi/Meetingbroker",
                "resources": "HSMAI certification, Cornell Hotel School online courses, STR Academy",
            },
            "marketing": {
                "focus": "campaign management, lead generation, content strategy, and marketing analytics",
                "tools": "Marketo, HubSpot, Google Analytics, SEMrush, Mailchimp",
                "resources": "HubSpot Academy, Google Digital Garage, 'Building a StoryBrand' by Donald Miller",
            },
            "finance": {
                "focus": "financial modeling, budgeting, forecasting, and regulatory compliance",
                "tools": "Excel, QuickBooks, SAP, Bloomberg Terminal, Tableau for finance",
                "resources": "Wall Street Prep, CFI (Corporate Finance Institute), CFA/CPA prep materials",
            },
            "hr": {
                "focus": "talent acquisition, employee lifecycle management, compliance, and people analytics",
                "tools": "Workday, BambooHR, LinkedIn Recruiter, Greenhouse, ADP",
                "resources": "SHRM certification, LinkedIn Learning HR courses, CIPD qualifications",
            },
        }
        
        ctx = domain_context.get(domain, domain_context.get("backend", {}))
        
        return {
            "reason": f"'{skill}' is specifically required in the {job_title} job description, indicating it's a key competency for this role. In the {domain_label} domain, this skill directly enables {ctx.get('focus', 'technical problem-solving and system design')}.",
            "timeline": "4-8 weeks depending on prior experience",
            "actions": [
                f"Research '{skill}' fundamentals: read official documentation and identify the top 3 sub-topics to master",
                f"Study through structured learning: {ctx.get('resources', 'Coursera, Udemy, or official documentation')}",
                f"Build a hands-on project applying '{skill}' using industry tools: {ctx.get('tools', 'relevant frameworks and platforms')}",
                f"Practice '{skill}' in the context of {domain_label}: focus on {ctx.get('focus', 'real-world application scenarios')}",
                f"Add '{skill}' proficiency to your portfolio with a documented project and measurable outcomes"
            ]
        }

    def _get_domain_specific_projects(self, domain, job_title, missing_skills, matched_skills):
        """Generate domain-specific project recommendations based on the detected domain."""
        
        projects_by_domain = {
            "frontend": [
                {
                    "name": "Responsive Analytics Dashboard with Real-Time Data Visualization",
                    "description": "Build a production-grade dashboard application featuring interactive charts, data tables with sorting/filtering, responsive layout, and real-time data updates via WebSockets. Implement authentication, dark/light theme toggling, and export functionality.",
                    "intuition": f"As a {job_title}, you'll be expected to build complex, data-rich user interfaces. This project demonstrates your ability to handle real-time data, responsive design, component architecture, and state management — all critical for frontend roles.",
                    "tech_stack": "React/Angular/Vue, TypeScript, D3.js or Chart.js, WebSocket, CSS Grid/Flexbox, Jest",
                },
                {
                    "name": "Accessible Component Library with Design System",
                    "description": "Create a reusable UI component library (buttons, modals, forms, tables, navigation) following WCAG 2.1 AA standards. Include Storybook documentation, unit tests, theming support, and publish as an npm package.",
                    "intuition": f"Companies hiring {job_title}s value engineers who understand design systems and accessibility. This project proves you can build scalable, reusable, and accessible UI components — a skill that sets you apart from most candidates.",
                    "tech_stack": "React/Angular, TypeScript, Storybook, CSS Modules/Tailwind, Jest, a11y testing tools",
                },
                {
                    "name": "Progressive Web App (PWA) — Task Management Platform",
                    "description": "Develop a full-featured task management PWA with offline support, push notifications, drag-and-drop kanban boards, real-time collaboration, and service worker caching strategies.",
                    "intuition": f"PWAs are increasingly demanded in frontend roles. This project demonstrates your ability to build performant, offline-capable web apps — key differentiator for {job_title} positions at modern companies.",
                    "tech_stack": "React/Vue, Service Workers, IndexedDB, Web Push API, Workbox, TypeScript",
                }
            ],
            "backend": [
                {
                    "name": "Scalable Microservices API Platform with Event-Driven Architecture",
                    "description": "Design and implement a microservices-based API system for an e-commerce domain with user service, product catalog, order management, and notification service. Include API gateway, message queues, distributed caching, and comprehensive API documentation.",
                    "intuition": f"Backend engineers must understand distributed systems. This project demonstrates your ability to design scalable architectures with proper service boundaries, async communication, and fault tolerance — exactly what {job_title} roles require.",
                    "tech_stack": "Node.js/Python, PostgreSQL, Redis, RabbitMQ/Kafka, Docker, Swagger/OpenAPI",
                },
                {
                    "name": "Real-Time Data Pipeline with Stream Processing",
                    "description": "Build an end-to-end data pipeline that ingests, transforms, and serves real-time data. Include a REST API layer, WebSocket push, database optimization, and monitoring/alerting.",
                    "intuition": f"Modern backend roles require data pipeline experience. This project shows you can handle high-throughput data flows, optimize database queries, and build reliable, observable systems.",
                    "tech_stack": "Python/Node.js, PostgreSQL, Redis Streams, Docker, Prometheus/Grafana",
                },
            ],
            "data_science": [
                {
                    "name": "End-to-End ML Pipeline: Customer Churn Prediction System",
                    "description": "Build a complete ML pipeline from data exploration to model deployment: EDA, feature engineering, model training (XGBoost, Random Forest, Neural Net), hyperparameter tuning, model evaluation with cross-validation, and REST API deployment with model versioning.",
                    "intuition": f"As a {job_title}, you need to demonstrate end-to-end ML lifecycle skills. This project covers the complete spectrum from raw data to production model — the most valued skill in data science hiring.",
                    "tech_stack": "Python, scikit-learn, XGBoost, pandas, MLflow, FastAPI, Docker",
                },
                {
                    "name": "NLP-Powered Sentiment Analysis & Topic Modeling Platform",
                    "description": "Create a text analytics platform that performs sentiment analysis, topic extraction, and trend detection on customer reviews or social media data. Include model fine-tuning, interactive visualizations, and an API.",
                    "intuition": f"NLP is one of the fastest-growing areas in data science. This project demonstrates advanced text processing skills and your ability to derive business insights from unstructured data.",
                    "tech_stack": "Python, Hugging Face Transformers, spaCy, NLTK, Streamlit, FastAPI",
                },
            ],
            "product_management": [
                {
                    "name": "Data-Driven Product Strategy Platform",
                    "description": "Build a product analytics dashboard that tracks user engagement metrics, feature adoption rates, funnel conversion, and A/B test results. Include automated weekly insight reports and a prioritization scoring engine using RICE framework.",
                    "intuition": f"As a {job_title}, you need to demonstrate data-fluency. This project shows you can define KPIs, instrument product analytics, run experiments, and use data to drive product decisions — the core of modern product management.",
                    "tech_stack": "Power BI/Tableau, SQL, Python, A/B testing framework, Jira API integration",
                },
                {
                    "name": "AI-Powered Product Roadmap & Feature Prioritization Tool",
                    "description": "Develop a tool that ingests customer feedback (surveys, support tickets, feature requests), uses NLP to cluster themes, auto-scores features by impact/effort, and generates a prioritized product roadmap with stakeholder views.",
                    "intuition": f"This bridges the gap between AI understanding and product sense. It demonstrates you can apply AI/ML to solve real product management challenges — making you stand out for {job_title} roles at tech companies.",
                    "tech_stack": "Python, NLP (topic modeling), Streamlit/React, SQL, Jira/Productboard API",
                },
                {
                    "name": "End-to-End Product Launch Case Study",
                    "description": "Document a complete product launch: market analysis, user personas, competitive analysis, PRD (Product Requirements Document), wireframes, go-to-market strategy, launch metrics, and post-launch iteration plan.",
                    "intuition": f"Product management is as much about communication as execution. This case study demonstrates your strategic thinking, stakeholder communication, and ability to drive a product from concept to market — essential for any {job_title}.",
                    "tech_stack": "Figma (wireframes), Notion/Confluence (documentation), Jira (backlog), analytics tools",
                }
            ],
            "ai_ml": [
                {
                    "name": "Intelligent Document Processing & Information Extraction System",
                    "description": "Build an AI system that automatically extracts structured information from unstructured documents (PDFs, images, forms) using OCR, NLP, and custom ML models. Include a feedback loop for model improvement and a REST API.",
                    "intuition": f"Document AI is a high-value enterprise use case. This project demonstrates end-to-end AI engineering skills: data processing, model training, API design, and productionization — all critical for {job_title} roles.",
                    "tech_stack": "Python, PyTorch/TensorFlow, Hugging Face, Tesseract/EasyOCR, FastAPI, Docker",
                },
                {
                    "name": "MLOps Platform: Automated Model Training & Monitoring Pipeline",
                    "description": "Create an automated ML pipeline that handles data ingestion, model training, evaluation, deployment, and monitoring with drift detection. Include experiment tracking, A/B model comparison, and automated retraining triggers.",
                    "intuition": f"MLOps is the #1 skill gap in AI teams. This project proves you can operationalize ML models at scale — transitioning from notebooks to production systems, which is exactly what senior {job_title} roles demand.",
                    "tech_stack": "Python, MLflow/Weights&Biases, Airflow, Docker, Kubernetes, Prometheus",
                }
            ],
            "devops": [
                {
                    "name": "Infrastructure-as-Code Multi-Environment Deployment Platform",
                    "description": "Design and implement a complete IaC setup with separate dev/staging/production environments, automated provisioning, CI/CD pipelines, monitoring, alerting, and disaster recovery procedures.",
                    "intuition": f"The core of DevOps is automating reliable infrastructure. This project demonstrates your ability to manage the full lifecycle of infrastructure deployment — the primary expectation for {job_title} positions.",
                    "tech_stack": "Terraform/Pulumi, Docker, Kubernetes, GitHub Actions/GitLab CI, Prometheus/Grafana",
                },
                {
                    "name": "Observability Stack: Centralized Logging, Metrics & Tracing",
                    "description": "Build a comprehensive observability platform with centralized logging (ELK/Loki), metrics collection (Prometheus), distributed tracing (Jaeger), and custom dashboards with alerting rules.",
                    "intuition": f"Observability is critical for production reliability. This project shows you understand the full monitoring spectrum — logs, metrics, and traces — which is essential for any {job_title} role.",
                    "tech_stack": "Prometheus, Grafana, Loki/Elasticsearch, Jaeger, Docker, Kubernetes",
                }
            ],
            "data_analytics": [
                {
                    "name": "Executive Business Intelligence Dashboard Suite",
                    "description": "Build a comprehensive BI dashboard suite covering sales, marketing, and operations KPIs with drill-through capabilities, automated data refresh, dynamic filters, forecasting visualizations, and scheduled email reports.",
                    "intuition": f"As a {job_title}, you'll be expected to turn raw data into actionable insights for executives. This project demonstrates your ability to design impactful visualizations, write complex queries, and deliver business value through analytics.",
                    "tech_stack": "Power BI/Tableau, SQL, Python (pandas), Excel, DAX/calculated fields",
                },
                {
                    "name": "Customer Segmentation & Cohort Analysis Platform",
                    "description": "Perform RFM segmentation, cohort retention analysis, and customer lifetime value modeling on real-world e-commerce data. Include interactive dashboards and an automated weekly report generator.",
                    "intuition": f"Customer analytics is the bread and butter of data analyst roles. This project proves you can derive meaningful business insights from complex datasets — the exact output hiring managers want to see from a {job_title}.",
                    "tech_stack": "SQL, Python (pandas, scipy), Power BI/Tableau, Jupyter Notebooks",
                }
            ],
            "cloud": [
                {
                    "name": "Cloud-Native Application Deployment & Auto-Scaling System",
                    "description": "Deploy a multi-tier application on cloud infrastructure with auto-scaling groups, load balancers, managed databases, CDN, and comprehensive IAM policies. Include cost optimization analysis and disaster recovery.",
                    "intuition": f"Cloud architecture is the core competency for {job_title} roles. This project demonstrates your ability to design, deploy, and manage production-grade cloud infrastructure with proper security and cost awareness.",
                    "tech_stack": "AWS/Azure/GCP, Terraform, Docker, Kubernetes, CloudWatch/Azure Monitor",
                }
            ],
            "sales": [
                {
                    "name": "B2B Sales Pipeline Optimization & CRM Dashboard",
                    "description": "Build a complete CRM-based sales pipeline with lead scoring, automated follow-up sequences, deal stage tracking, and revenue forecasting dashboards. Include weekly/monthly sales reports with win-rate analysis and conversion metrics.",
                    "intuition": f"As a {job_title}, your ability to manage and optimize a sales pipeline is directly tied to quota attainment. This project demonstrates you can set up and run a professional sales operation from prospecting to close.",
                    "tech_stack": "Salesforce/HubSpot/Pipedrive CRM, Excel (pivot tables, charts), PowerPoint (QBR decks)",
                },
                {
                    "name": "Corporate Account Acquisition Strategy & Pitch Deck",
                    "description": "Develop a complete corporate account acquisition playbook: target account identification, competitive analysis, value proposition development, customized pitch decks, rate proposal templates, and objection handling scripts.",
                    "intuition": f"B2B account acquisition is the core revenue driver for {job_title} roles. This project shows you can research prospects, craft compelling proposals, and execute a systematic approach to winning new corporate business.",
                    "tech_stack": "PowerPoint, Excel, LinkedIn Sales Navigator, CRM system, market research tools",
                },
                {
                    "name": "Sales Performance Analytics & Forecasting Model",
                    "description": "Create a comprehensive sales analytics system: track KPIs (conversion rate, average deal size, sales cycle length), build revenue forecasting models, and generate automated weekly reports with trend analysis.",
                    "intuition": f"Data-driven sales management separates top performers from average sellers. This project demonstrates you understand sales metrics, can forecast accurately, and make data-backed decisions to optimize your {job_title} performance.",
                    "tech_stack": "Excel (advanced formulas, pivot tables), CRM reporting, Tableau/Power BI (optional)",
                }
            ],
            "hospitality": [
                {
                    "name": "Hotel Corporate Rate Negotiation & RFP Response System",
                    "description": "Develop a complete RFP response system for hotel corporate rates: competitive rate analysis, proposal templates for different account tiers (top 50/100/500 companies), contract term sheets, and a tracking system for rate loading across GDS channels.",
                    "intuition": f"Corporate rate management is the lifeblood of hotel B2B sales. As a {job_title}, this project proves you understand the full RFP lifecycle — from market analysis to rate proposal to contract execution — and can manage a corporate rate portfolio systematically.",
                    "tech_stack": "Excel (rate comparison models), PowerPoint (proposals), Amadeus/Sabre GDS, Opera PMS",
                },
                {
                    "name": "Travel Management Company (TMC) Partnership Playbook",
                    "description": "Create a comprehensive TMC partnership strategy: identify top TMCs in your market, develop customized partnership proposals, create commission structures, build a tracking system for TMC production, and design quarterly business review templates.",
                    "intuition": f"TMC relationships are a key revenue channel for hotel sales. This project demonstrates you understand the TMC ecosystem, can build strategic partnerships, and manage channel performance — critical skills for a {job_title}.",
                    "tech_stack": "CRM, Excel (production tracking), PowerPoint (QBR decks), Amadeus/Sabre connection reports",
                },
                {
                    "name": "Market Competitive Analysis & Revenue Strategy Dashboard",
                    "description": "Build a market intelligence system: collect and analyze competitor rates (via STR reports), track local market demand drivers (events, seasonality), create pricing strategy recommendations, and present findings in executive-ready dashboard format.",
                    "intuition": f"Understanding your competitive set and market dynamics is essential for any {job_title}. This project shows you can translate market data into pricing decisions that maximize RevPAR and market share.",
                    "tech_stack": "STR reports, Excel (comp set analysis), PowerPoint, Tableau/Power BI (optional)",
                }
            ],
            "marketing": [
                {
                    "name": "Multi-Channel Marketing Campaign with ROI Tracking",
                    "description": "Design and execute a multi-channel marketing campaign (email, social media, content, paid ads) with complete tracking: CAC, ROAS, conversion rates, and attribution modeling. Include A/B test results and optimization recommendations.",
                    "intuition": f"As a {job_title}, you need to demonstrate ROI for marketing spend. This project proves you can plan, execute, and measure campaigns across channels — the core competency hiring managers evaluate.",
                    "tech_stack": "Google Analytics, HubSpot/Marketo, Google Ads, Social media platforms, Excel",
                }
            ],
        }
        
        projects = projects_by_domain.get(domain, [])
        
        # Add skills_covered to each project
        for proj in projects:
            if "skills_covered" not in proj:
                proj["skills_covered"] = missing_skills[:3]
        
        # If no domain-specific projects, generate from knowledge base
        if not projects:
            return []
        
        return projects[:3]
    
    def _get_domain_specific_action_plan(self, domain, job_title, missing_skills, matched_skills,
                                          projects, future_paths, skill_contexts):
        """Generate a detailed, domain-specific 90-day action plan."""
        
        # Build skill-specific study topics from contexts
        study_topics = []
        practice_items = []
        for skill in missing_skills[:3]:
            ctx = next((c for c in skill_contexts if c.get("skill", "").lower() == skill.lower()), None)
            if ctx:
                resources = ctx.get("learning_resources", [])
                for r in resources[:2]:
                    study_topics.append(r)
                project_ideas = ctx.get("project_ideas", [])
                for p in project_ideas[:1]:
                    practice_items.append(p)
        
        skill_names = [s.title() for s in missing_skills[:3]]
        skill_text = ", ".join(skill_names) if skill_names else "core role requirements"
        proj_name = projects[0]["name"] if projects else "your portfolio project"
        proj2_name = projects[1]["name"] if len(projects) > 1 else "your second project"
        
        # Domain-specific action plans
        domain_plans = {
            "frontend": {
                "weeks_1_4": [
                    f"**Core Concepts (10-12 hrs/week):** Deep-dive into {skill_text}. Study component lifecycle, hooks/directives, state management patterns, and routing.",
                    f"**Specific Topics to Master:** Virtual DOM reconciliation, unidirectional data flow, CSS-in-JS vs CSS Modules, responsive design breakpoints, Flexbox/Grid mastery.",
                    f"**Practice:** Complete 3 coding challenges daily on Frontend Mentor or CSS Battle. Build 5 small UI components (modal, dropdown, carousel, data table, infinite scroll).",
                    f"**Resources:** {study_topics[0] if study_topics else 'Complete the official documentation tutorials for your target framework.'}",
                    f"**Concentrate On:** Understanding WHY patterns exist (e.g., why immutability matters for rendering, why unidirectional data flow prevents bugs). Build mental models, not just syntax."
                ],
                "weeks_5_8": [
                    f"**Build '{proj_name}':** Start with wireframes → component hierarchy → implement core features. Focus on: proper folder structure, reusable components, custom hooks/services, error boundaries.",
                    f"**Technical Depth:** Implement performance optimization (React.memo / OnPush / computed), code splitting, lazy loading, and proper SEO meta tags. Add comprehensive unit tests (target 75%+ coverage).",
                    f"**Build '{proj2_name}':** Focus on accessibility (ARIA labels, keyboard navigation, screen reader testing), internationalization, and cross-browser compatibility.",
                    f"**Practice:** {practice_items[0] if practice_items else 'Recreate a popular website UI (e.g., Spotify, Twitter) pixel-perfect with responsive design.'}",
                    f"**Concentrate On:** Code quality — consistent naming, proper TypeScript types, meaningful component abstractions, and clean git commit history."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Add projects with quantified metrics: 'Built responsive dashboard reducing page load from 3.2s to 1.1s' or 'Achieved 95+ Lighthouse score'. List specific technologies used.",
                    f"**Interview Prep:** Practice system design for frontend: 'Design a real-time collaborative editor', 'Design an infinite scroll feed', 'Design a design system'. Study performance optimization patterns.",
                    f"**Coding Interview Practice:** Solve 30+ DOM manipulation, async JS, and React/framework-specific problems. Practice explaining your code architecture decisions out loud.",
                    f"**Network & Apply:** Apply for {future_paths[0] if future_paths else job_title} roles, contribute to open-source frontend projects, attend frontend meetups/conferences.",
                    f"**Concentrate On:** Being able to articulate architectural decisions in interviews — WHY you chose a state management library, HOW you'd optimize a slow component, WHAT trade-offs you considered."
                ]
            },
            "backend": {
                "weeks_1_4": [
                    f"**Core Concepts (10-12 hrs/week):** Master {skill_text}. Study REST API design principles, database modeling (normalization, indexing, query optimization), and authentication patterns (JWT, OAuth 2.0).",
                    f"**Specific Topics:** SOLID principles, design patterns (Repository, Factory, Observer), HTTP status codes, request validation, error handling middleware, database transactions and ACID properties.",
                    f"**Practice:** Build 3 small APIs from scratch (CRUD operations with validation, auth, pagination). Write SQL queries with JOINs, subqueries, CTEs, and window functions daily.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Complete the official framework documentation and build along with tutorials.'}",
                    f"**Concentrate On:** Understanding distributed systems fundamentals — CAP theorem, eventual consistency, message queues, caching strategies. Think about SCALE from day one."
                ],
                "weeks_5_8": [
                    f"**Build '{proj_name}':** Implement rate limiting, input validation, proper error handling, database migrations, connection pooling, and comprehensive logging.",
                    f"**Technical Depth:** Add Redis caching layer, implement background job processing, set up database indexing strategy, write integration tests with test containers.",
                    f"**Deployment:** Containerize with Docker, set up CI/CD pipeline, implement health checks, configure environment variables securely.",
                    f"**Concentrate On:** Production readiness — proper logging, monitoring, graceful shutdown, database connection management. These separate junior from senior developers."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Quantify: 'Designed API handling 1000+ req/sec with p99 latency < 200ms', 'Reduced query time by 80% through indexing strategy'. Highlight system design thinking.",
                    f"**Interview Prep:** Practice system design: 'Design a URL shortener', 'Design a rate limiter', 'Design a message queue'. Study common backend interview topics: concurrency, caching, database sharding.",
                    f"**Coding Practice:** Solve 40+ LeetCode problems focused on arrays, hash maps, trees, and graph traversal. Practice writing clean, production-quality code.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles. Prepare to discuss your project architecture decisions, trade-offs, and scaling strategies."
                ]
            },
            "product_management": {
                "weeks_1_4": [
                    f"**Core Concepts (8-10 hrs/week):** Deep-dive into {skill_text}. Study product strategy frameworks: Jobs-to-be-Done (JTBD), Design Thinking, Lean Startup methodology, and product-market fit.",
                    f"**Specific Topics to Master:** RICE/ICE prioritization, North Star Metric framework, user story writing (INVEST criteria), product requirement documents (PRDs), competitive analysis templates.",
                    f"**Practice:** Write 5 PRDs for real products. Define KPIs and OKRs for 3 different product scenarios. Conduct 3 mock user interviews.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Read Inspired by Marty Cagan, The Lean Product Playbook by Dan Olsen, and Continuous Discovery Habits by Teresa Torres.'}",
                    f"**Concentrate On:** Understanding the WHY behind product decisions. Practice translating user pain points into measurable product outcomes. Think like a CEO of the product."
                ],
                "weeks_5_8": [
                    f"**Build '{proj_name}':** Create a complete product strategy deck: market analysis, user personas, competitive landscape, feature prioritization matrix, and product roadmap with quarterly objectives.",
                    f"**Hands-On Analytics:** Set up product analytics tracking, define conversion funnels, run A/B test analyses, build KPI dashboards in Power BI/Tableau.",
                    f"**Stakeholder Skills:** Practice presenting product strategy to mock stakeholder groups. Create executive summaries, sprint review decks, and quarterly business review (QBR) templates.",
                    f"**Concentrate On:** Data storytelling — being able to extract insights from metrics and communicate them to different audiences (engineers, executives, customers)."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Quantify impact: 'Led product initiative increasing user retention by 25%', 'Defined product roadmap adopted by 3 cross-functional teams'. Use metrics-driven bullet points.",
                    f"**Interview Prep:** Practice product sense interviews: 'How would you improve Instagram Stories?', 'Design a product for elderly users'. Practice analytical cases with metrics estimation.",
                    f"**Case Studies:** Prepare 3 deep-dive case studies from your projects demonstrating end-to-end product thinking. Practice presenting them in 15-minute walk-throughs.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles. Network with PMs through Product School, Mind the Product, Lenny's Newsletter communities."
                ]
            },
            "data_analytics": {
                "weeks_1_4": [
                    f"**Core Concepts (10-12 hrs/week):** Master {skill_text}. Study SQL deeply: JOINs, subqueries, CTEs, window functions (ROW_NUMBER, RANK, LAG/LEAD), and query optimization.",
                    f"**Specific Topics:** Statistical fundamentals (distributions, hypothesis testing, confidence intervals, correlation vs causation), data cleaning techniques, ETL concepts.",
                    f"**Practice:** Solve 5 SQL challenges daily on LeetCode/HackerRank. Build 3 small dashboards from public datasets (Kaggle). Master Excel pivot tables and VLOOKUP/INDEX-MATCH.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Complete Google Data Analytics Professional Certificate on Coursera.'}",
                    f"**Concentrate On:** Thinking in terms of BUSINESS QUESTIONS, not just queries. For every analysis ask: 'What decision will this inform? Who is the audience? What action should follow?'"
                ],
                "weeks_5_8": [
                    f"**Build '{proj_name}':** Create interactive dashboards with drill-through navigation, dynamic date filters, calculated measures (YoY growth, running averages, percentage of total).",
                    f"**Technical Depth:** Perform cohort analysis, RFM segmentation, funnel analysis, and A/B test statistical analysis on real datasets. Document methodology and insights.",
                    f"**Python/R:** Learn pandas for data manipulation (groupby, merge, pivot_table, apply). Create automated reports with matplotlib/seaborn visualizations.",
                    f"**Concentrate On:** Presentation quality — insights must be actionable. Every chart needs a 'so what?' takeaway. Practice explaining complex analysis simply to non-technical stakeholders."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Add metrics: 'Identified $500K revenue opportunity through customer segmentation analysis', 'Reduced reporting time by 60% through Power BI automation'. Showcase dashboards in portfolio.",
                    f"**Interview Prep:** Practice SQL live coding interviews (30-minute query problems). Prepare for case study presentations: 'Given this dataset, what insights would you present to the VP of Marketing?'",
                    f"**Portfolio:** Create a GitHub Pages portfolio showcasing 3-4 analysis projects with clean visualizations, methodology documentation, and business recommendations.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles. Prepare to discuss your analytical process, tool preferences, and how you handle ambiguous data requests."
                ]
            },
            "ai_ml": {
                "weeks_1_4": [
                    f"**Core Concepts (12-15 hrs/week):** Master {skill_text}. Study supervised/unsupervised learning algorithms, neural network architectures, loss functions, optimizers, and regularization techniques.",
                    f"**Specific Topics:** Linear/Logistic regression internals, Decision Trees/Random Forest/XGBoost, CNNs, RNNs/LSTMs, Transformers architecture, attention mechanism, batch normalization, dropout.",
                    f"**Practice:** Complete 5 Kaggle notebooks. Implement algorithms from scratch in NumPy before using libraries. Train and evaluate models with proper cross-validation and hyperparameter tuning.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Complete Andrew Ng ML Specialization + Deep Learning Specialization on Coursera.'}",
                    f"**Concentrate On:** Mathematical intuition behind algorithms — understand gradient descent, backpropagation, and why specific architectures work for specific problem types."
                ],
                "weeks_5_8": [
                    f"**Build '{proj_name}':** Full pipeline: data collection → EDA → feature engineering → model selection → training → evaluation → deployment. Use proper experiment tracking (MLflow/W&B).",
                    f"**Technical Depth:** Implement model versioning, A/B testing between models, monitoring for data/concept drift, and automated retraining pipeline.",
                    f"**MLOps Practice:** Containerize models with Docker, create inference API with FastAPI, set up model registry, implement CI/CD for ML pipelines.",
                    f"**Concentrate On:** Production ML vs. notebook ML. Focus on reproducibility, scalability, monitoring, and maintaining models in production."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Quantify: 'Deployed customer churn model achieving 92% AUC, saving $300K annually', 'Reduced inference latency from 500ms to 50ms through model optimization'. Include GitHub links to projects.",
                    f"**Interview Prep:** Practice ML system design: 'Design a recommendation system for Netflix', 'Design a fraud detection system at scale'. Review ML theory: bias-variance tradeoff, overfitting, feature selection.",
                    f"**Coding Practice:** Implement common ML algorithms from scratch. Practice coding interviews with Python — focus on data manipulation and algorithm implementation.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles. Prepare to walk through your entire ML pipeline and discuss trade-offs at each stage."
                ]
            },
            "devops": {
                "weeks_1_4": [
                    f"**Core Concepts (10-12 hrs/week):** Master {skill_text}. Study Linux administration, networking (TCP/IP, DNS, load balancing), containerization fundamentals, and cloud service models.",
                    f"**Specific Topics:** Docker internals (namespaces, cgroups), Kubernetes architecture (pods, services, deployments, ingress), Terraform HCL syntax, CI/CD pipeline design patterns.",
                    f"**Practice:** Set up a local Kubernetes cluster (Minikube/Kind). Write Dockerfiles for 5 different application types. Create Terraform modules for common infrastructure patterns.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Study for CKA (Certified Kubernetes Administrator) or AWS Solutions Architect Associate.'}",
                    f"**Concentrate On:** Understanding the 'why' behind DevOps practices — immutable infrastructure, GitOps, infrastructure as code. These principles matter more than specific tool syntax."
                ],
                "weeks_5_8": [
                    f"**Build '{proj_name}':** Design complete multi-environment infrastructure with proper networking, security groups, IAM policies, and secret management.",
                    f"**CI/CD Pipeline:** Build automated pipeline: lint → test → build → security scan → deploy to staging → smoke test → promote to production. Include rollback strategies.",
                    f"**Monitoring Stack:** Set up Prometheus + Grafana dashboards, alerting rules, log aggregation, and distributed tracing. Define SLIs/SLOs for your applications.",
                    f"**Concentrate On:** Security best practices — least privilege IAM, secret rotation, network segmentation, container vulnerability scanning. Security is DevOps differentiation."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Quantify: 'Reduced deployment time from 2 hours to 15 minutes through CI/CD automation', 'Achieved 99.9% uptime SLA through automated scaling and monitoring'. List certifications.",
                    f"**Interview Prep:** Practice infrastructure design interviews: 'Design the infrastructure for a high-traffic web application', 'How would you handle a production outage?'. Review troubleshooting scenarios.",
                    f"**Certifications:** Complete CKA, AWS SA Associate, or Terraform Associate certification exam. These significantly boost DevOps candidacy.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles. Be prepared to whiteboard infrastructure diagrams and discuss trade-offs between cloud services."
                ]
            },
            "sales": {
                "weeks_1_4": [
                    f"**Core Sales Skills (8-10 hrs/week):** Master {skill_text}. Study consultative selling frameworks: SPIN Selling, Challenger Sale, and Solution Selling methodologies.",
                    f"**CRM Proficiency:** Complete Salesforce Trailhead or HubSpot Academy free certification. Set up a practice pipeline with proper stages: Lead → Qualified → Proposal → Negotiation → Won/Lost.",
                    f"**Cold Outreach Mastery:** Script and practice 5 different cold call openers. Write 3 cold email templates with A/B testing variants. Practice handling the top 10 sales objections.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Read SPIN Selling by Neil Rackham and Never Split the Difference by Chris Voss.'}",
                    f"**Concentrate On:** Understanding buyer psychology — why people buy, how decisions are made in B2B organizations (champion vs. decision-maker), and how to build urgency without being pushy."
                ],
                "weeks_5_8": [
                    f"**Build Your Sales Playbook:** Create a complete account acquisition strategy: target account list (50 companies), competitive battle cards, value proposition for each buyer persona, and presentation templates.",
                    f"**Negotiation Practice:** Role-play 10 negotiation scenarios with a partner. Practice BATNA analysis, anchoring techniques, and creating win-win outcomes. Document your negotiation scripts.",
                    f"**Pipeline Development:** Build a prospecting system: LinkedIn Sales Navigator research → personalized outreach → follow-up cadence → meeting booking → proposal delivery. Track all activities in CRM.",
                    f"**Concentrate On:** Developing a repeatable sales process. Top performers don't wing it — they follow a systematic approach from prospecting to close with metrics at every stage."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Quantify achievements: 'Generated $500K in pipeline through cold outreach', 'Achieved 120% quota attainment in Q3', 'Closed 25 new corporate accounts in 6 months'. Use revenue/percentage metrics.",
                    f"**Interview Prep:** Prepare for behavioral interviews: 'Tell me about a time you lost a deal and what you learned', 'How do you handle a prospect who goes silent?', 'Walk me through your sales process'. Practice STAR format responses.",
                    f"**Mock Sales Presentations:** Record yourself delivering 3 different pitch presentations. Review for pacing, confidence, and objection handling. Get feedback from peers or mentors.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles. Network at industry events, connect with hiring managers on LinkedIn, and prepare a strong 60-second elevator pitch about your sales approach."
                ]
            },
            "hospitality": {
                "weeks_1_4": [
                    f"**Hospitality Sales Foundations (10-12 hrs/week):** Master {skill_text}. Study hotel revenue management: ADR, RevPAR, occupancy rates, rate parity, and yield management principles.",
                    f"**Industry Knowledge:** Learn the hotel distribution ecosystem: GDS (Amadeus, Sabre, Travelport), OTAs (Booking.com, Expedia), TMC partnerships, and direct booking channels. Understand how corporate rates are structured (LRA, dynamic, static).",
                    f"**CRM & Tools Mastery:** Complete training on hospitality CRM systems (Delphi, Opera Sales & Catering). Master travel booking platforms (Amadeus, Concur Travel). Build Excel templates for rate analysis and production tracking.",
                    f"**Resources:** {study_topics[0] if study_topics else 'Join HSMAI (Hospitality Sales & Marketing Association International) and complete their CHDM certificate prep materials. Study STR reports for competitive analysis.'}",
                    f"**Concentrate On:** Understanding the corporate travel buying cycle: RFP season (Aug-Oct), rate loading deadlines, and how travel managers make hotel selection decisions. This seasonal knowledge is critical."
                ],
                "weeks_5_8": [
                    f"**Corporate Sales Execution:** Develop a target account list of 50 companies in your market. Research their travel patterns, current hotel partners, and decision-makers. Create customized rate proposals for each tier.",
                    f"**Relationship Building:** Practice property site inspections and client entertainment. Create a compelling hotel tour script that highlights differentiators. Build a client event strategy (FAM trips, networking dinners, industry mixers).",
                    f"**Rate Negotiation Practice:** Role-play 10 corporate rate negotiation scenarios: RFP response, counter-offer from procurement, volume commitment discussions, and contract renewal negotiations. Document your negotiation playbook.",
                    f"**Concentrate On:** Becoming a trusted advisor, not just a salesperson. Corporate travel buyers want partners who understand their travel policy, can solve problems proactively, and deliver consistent service across stays."
                ],
                "weeks_9_12": [
                    f"**Resume Polish:** Quantify: 'Grew corporate room night production by 35% ($180K incremental revenue)', 'Secured 15 new negotiated corporate accounts', 'Increased ADR by $12 through strategic rate positioning'. Use hospitality-specific metrics.",
                    f"**Interview Prep:** Prepare for hospitality sales interviews: 'How would you win a corporate account from a competitor?', 'Walk me through your RFP process', 'How do you handle a client threatening to move their business?'. Research each hotel's competitive position.",
                    f"**Industry Networking:** Attend HSMAI events, local hotel association meetings, and corporate travel buyer conferences. Build relationships with TMC account managers and corporate travel coordinators.",
                    f"**Apply:** Target {future_paths[0] if future_paths else job_title} roles at hotels in your preferred market. Prepare a 90-day business plan as part of your interview — this dramatically differentiates you from other candidates."
                ]
            },
        }
        
        # Get domain-specific plan or build a generic detailed plan
        plan = domain_plans.get(domain, None)
        
        if plan:
            return plan
        
        # Fallback: Build detailed generic plan using skill context
        return {
            "weeks_1_4": [
                f"**Core Skill Study (10-12 hrs/week):** Focus on mastering fundamentals of {skill_text}. Study official documentation and complete structured courses.",
                f"**Specific Topics:** {'; '.join(study_topics[:3]) if study_topics else f'Core concepts, best practices, and common patterns for {skill_text}'}",
                f"**Daily Practice:** Complete 2-3 hands-on exercises or coding challenges related to {missing_skills[0].title() if missing_skills else 'your target skills'} every day.",
                f"**Concentrate On:** Building deep understanding of core concepts. Don't rush through tutorials — implement everything from scratch and debug intentionally."
            ],
            "weeks_5_8": [
                f"**Build '{proj_name}':** Start implementation focusing on core features first. Follow best practices for the technology stack.",
                f"**Technical Depth:** {practice_items[0] if practice_items else f'Build increasingly complex features incorporating {skill_text}'}. Add testing, documentation, and proper error handling.",
                f"**Code Quality:** Implement CI/CD, write documentation, and ensure your code follows industry standards. Get feedback from peers or mentors.",
                f"**Concentrate On:** Shipping working software. A completed project with good documentation beats 5 half-finished projects."
            ],
            "weeks_9_12": [
                f"**Resume Optimization:** Add quantified achievements from your projects. Use the format: 'Action + Technology + Measurable Result'. Include links to live projects and GitHub repos.",
                f"**Interview Preparation:** Practice common interview questions for {job_title} roles. Prepare 3 project deep-dives covering your technical decisions and trade-offs.",
                f"**Apply & Network:** Apply for {future_paths[0] if future_paths else job_title} roles. Attend industry meetups, contribute to open-source, and build your professional network on LinkedIn.",
                f"**Concentrate On:** Articulating your learning journey and project decisions. Interviewers value candidates who can explain 'why' behind their choices."
            ]
        }


def get_llm_client(provider=None):
    """
    Get LLM client instance.
    
    Args:
        provider: "openai" or "gemini" (auto-detects from env if not specified)
    
    Returns:
        LLMClient instance or None if no API key available
    """
    if provider is None:
        # Auto-detect based on available API keys
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            print("⚠️  No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY environment variable.")
            print("    Career advice generation will be disabled.")
            return None
    
    try:
        return LLMClient(provider=provider)
    except Exception as e:
        print(f"⚠️  Failed to initialize LLM client: {e}")
        return None
