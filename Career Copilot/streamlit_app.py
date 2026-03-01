"""
Streamlit UI for Career Readiness System
User-friendly interface for skill gap analysis and career guidance.
"""

import streamlit as st
import requests
from io import BytesIO
import json

# Page configuration
st.set_page_config(
    page_title="Career Copilot - AI Career Readiness",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .skill-matched {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .skill-missing {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚀 Career Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Career Readiness & Skill Gap Analysis</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_endpoint = st.text_input(
        "API Endpoint",
        value="http://localhost:8000",
        help="FastAPI backend URL"
    )
    
    use_llm = st.checkbox(
        "Generate AI Career Advice",
        value=True,
        help="Requires LLM API key (OpenAI or Gemini)"
    )
    
    st.divider()
    
    st.markdown("""
    ### 📊 How it works
    
    1. **Upload** your resume (PDF)
    2. **Paste** job description
    3. **Analyze** skill gaps
    4. **Get** personalized career advice
    
    ### 🔑 Features
    - Semantic skill matching
    - Abbreviation handling (AI, ML, KPI)
    - RAG-powered recommendations
    - LLM-generated career roadmap
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📄 Analysis", "📚 Knowledge Base", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload Resume")
        resume_file = st.file_uploader(
            "Upload PDF Resume",
            type=["pdf"],
            help="Select your resume in PDF format"
        )
    
    with col2:
        st.subheader("📋 Job Description")
        job_description = st.text_area(
            "Paste Job Description",
            height=200,
            help="Copy and paste the complete job description"
        )
    
    # Analyze button
    if st.button("🔍 Analyze Skills", type="primary", use_container_width=True):
        if not resume_file:
            st.error("⚠️ Please upload a resume")
        elif not job_description:
            st.error("⚠️ Please provide a job description")
        else:
            with st.spinner("🔄 Analyzing your skills..."):
                try:
                    # Call API
                    files = {"resume": ("resume.pdf", resume_file, "application/pdf")}
                    data = {"job_description": job_description}
                    
                    response = requests.post(
                        f"{api_endpoint}/api/analyze",
                        files=files,
                        data=data,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display metrics
                        st.success("✅ Analysis Complete!")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric(
                                "Match Score",
                                f"{result['match_score']}%",
                                delta=None
                            )
                        
                        with metric_col2:
                            st.metric(
                                "JD Skills",
                                result['jd_skills_count']
                            )
                        
                        with metric_col3:
                            st.metric(
                                "✅ Matched",
                                len(result['matched_skills']),
                                delta="positive"
                            )
                        
                        with metric_col4:
                            st.metric(
                                "❌ Missing",
                                len(result['missing_skills']),
                                delta="negative"
                            )
                        
                        st.divider()
                        
                        # Display skills
                        col_matched, col_missing = st.columns(2)
                        
                        with col_matched:
                            st.subheader(f"✅ Matched Skills ({len(result['matched_skills'])})")
                            for skill in result['matched_skills'][:20]:
                                st.markdown(f'<div class="skill-matched">👍 {skill}</div>', unsafe_allow_html=True)
                            if len(result['matched_skills']) > 20:
                                with st.expander("Show all matched skills"):
                                    for skill in result['matched_skills'][20:]:
                                        st.markdown(f"• {skill}")
                        
                        with col_missing:
                            st.subheader(f"❌ Missing Skills ({len(result['missing_skills'])})")
                            for skill in result['missing_skills']:
                                st.markdown(f'<div class="skill-missing">📌 {skill}</div>', unsafe_allow_html=True)
                        
                        # Career advice
                        if use_llm and result.get('career_advice'):
                            st.divider()
                            st.header("🎯 AI-Generated Career Guidance")
                            
                            advice = result['career_advice']

                            if advice.get('error'):
                                st.warning(f"⚠️ {advice.get('error', 'Career guidance used fallback mode.')}" )

                            # Career Summary
                            st.subheader("📊 Career Readiness Summary")
                            st.info(advice.get('career_summary', 'N/A'))
                            
                            # Strengths
                            if advice.get('strengths'):
                                st.subheader("💪 Your Strengths")
                                for strength in advice['strengths']:
                                    st.markdown(f"✓ {strength}")
                            
                            # Priority Skills
                            if advice.get('priority_skills'):
                                st.subheader("🎯 Priority Skill Development")
                                for idx, skill in enumerate(advice['priority_skills'], 1):
                                    skill_name = skill.get('skill', 'N/A')
                                    timeline = skill.get('timeline', '')
                                    with st.expander(f"🔹 #{idx} — {skill_name}  ⏱️ {timeline}"):
                                        reason = skill.get('reason', '')
                                        if reason and reason != 'N/A':
                                            st.markdown(f"**💡 Why it matters:** {reason}")
                                        st.markdown("---")
                                        if skill.get('actions'):
                                            st.markdown("**📋 Action Steps to Master This Skill:**")
                                            for i, action in enumerate(skill['actions'], 1):
                                                st.markdown(f"{i}. {action}")
                            
                            # Projects
                            if advice.get('recommended_projects'):
                                st.subheader("🚀 Recommended Projects")
                                for project in advice['recommended_projects']:
                                    with st.expander(f"📁 {project.get('name', 'Project')}"):
                                        st.markdown(f"**📋 Description:** {project.get('description', 'N/A')}")
                                        if project.get('intuition'):
                                            st.markdown(f"**💡 Why This Project:** {project['intuition']}")
                                        if project.get('tech_stack'):
                                            st.markdown(f"**🛠️ Tech Stack:** {project['tech_stack']}")
                                        if project.get('skills_covered'):
                                            st.markdown("**🎯 Skills Covered:** " + ", ".join(project['skills_covered']))
                            
                            # Career Paths
                            if advice.get('career_paths'):
                                st.subheader("🎯 Career Path Recommendations")
                                col_immediate, col_future = st.columns(2)
                                
                                with col_immediate:
                                    st.markdown("**Immediate Opportunities:**")
                                    for path in advice['career_paths'].get('immediate', []):
                                        st.markdown(f"• {path}")
                                
                                with col_future:
                                    st.markdown("**After Upskilling:**")
                                    for path in advice['career_paths'].get('after_upskilling', []):
                                        st.markdown(f"• {path}")
                            
                            # 90-Day Action Plan
                            if advice.get('action_plan'):
                                st.subheader("📅 90-Day Action Plan")
                                plan = advice['action_plan']
                                
                                col_w1, col_w2, col_w3 = st.columns(3)
                                
                                with col_w1:
                                    st.markdown("#### 📖 Weeks 1-4: Deep Learning Phase")
                                    for item in plan.get('weeks_1_4', []):
                                        st.markdown(f"• {item}")
                                
                                with col_w2:
                                    st.markdown("#### 🔨 Weeks 5-8: Build & Practice Phase")
                                    for item in plan.get('weeks_5_8', []):
                                        st.markdown(f"• {item}")
                                
                                with col_w3:
                                    st.markdown("#### 🚀 Weeks 9-12: Polish & Apply Phase")
                                    for item in plan.get('weeks_9_12', []):
                                        st.markdown(f"• {item}")
                    else:
                        st.error(f"❌ Analysis failed: {response.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API server. Make sure FastAPI is running at " + api_endpoint)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab2:
    st.header("📚 Skill Knowledge Base")
    st.markdown("""
    Search for information about specific skills, including learning resources,
    project ideas, and career paths.
    """)
    
    skill_query = st.text_input("🔍 Search for a skill", placeholder="e.g., machine learning, product management")
    
    if skill_query and st.button("Search"):
        try:
            response = requests.post(
                f"{api_endpoint}/api/skill-context",
                params={"skill": skill_query},
                timeout=10
            )
            
            if response.status_code == 200:
                context = response.json()
                
                st.subheader(f"📖 {context['skill'].title()}")
                st.info(context['description'])
                
                col_learn, col_proj = st.columns(2)
                
                with col_learn:
                    st.markdown("**📚 Learning Resources:**")
                    for resource in context['learning_resources']:
                        st.markdown(f"• {resource}")
                
                with col_proj:
                    st.markdown("**🚀 Project Ideas:**")
                    for project in context['project_ideas']:
                        st.markdown(f"• {project}")
                
                st.markdown(f"**⏱️ Estimated Time:** {context['estimated_time']}")
                st.markdown(f"**💼 Career Paths:** {', '.join(context['career_paths'])}")
            else:
                st.error("Skill not found in knowledge base")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab3:
    st.header("ℹ️ About Career Copilot")
    
    st.markdown("""
    ### 🎯 Mission
    Career Copilot is an AI-powered career readiness system that helps job seekers:
    - Understand their skill gaps
    - Get personalized learning recommendations
    - Create actionable career development plans
    
    ### 🛠️ Technology Stack
    
    **Backend:**
    - 🔷 FastAPI for REST API
    - 🧠 Sentence Transformers for semantic embeddings
    - 🔍 FAISS for efficient vector search (RAG)
    - 🤖 OpenAI GPT / Google Gemini for career advice generation
    
    **Frontend:**
    - 🎨 Streamlit for interactive UI
    
    **AI/ML:**
    - Semantic skill matching with cosine similarity
    - Abbreviation handling (AI ↔ artificial intelligence)
    - RAG (Retrieval-Augmented Generation) for grounded advice
    - LLM-powered career guidance
    
    ### 📊 How It Works
    
    1. **Resume Ingestion**: Extract text from PDF using PDFPlumber
    2. **Preprocessing**: Clean text and segment into sentences
    3. **Skill Extraction**: NLP-based phrase extraction with spaCy
    4. **Semantic Analysis**: Generate embeddings and compute similarity
    5. **RAG**: Retrieve relevant context from knowledge base
    6. **LLM Generation**: Create personalized career advice
    
    ### 👨‍💻 Developer
    Built with ❤️ using modern AI/ML technologies
    
    ### 📝 License
    Educational and research purposes
    """)
    
    # API Health Check
    st.divider()
    st.subheader("🔧 System Status")
    
    try:
        response = requests.get(f"{api_endpoint}/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "🟢 Online" if health['api'] == "healthy" else "🔴 Offline"
                st.metric("API Status", status)
            
            with col2:
                rag_status = "🟢 Available" if health['rag_engine'] == "available" else "🔴 Unavailable"
                st.metric("RAG Engine", rag_status)
            
            with col3:
                llm_status = "🟢 Available" if health['llm_client'] == "available" else "🔴 Not Configured"
                st.metric("LLM Client", llm_status)
                if health.get('llm_provider'):
                    st.caption(f"Provider: {health['llm_provider']}")
        else:
            st.error("API health check failed")
    except:
        st.error("❌ Cannot connect to API server")
