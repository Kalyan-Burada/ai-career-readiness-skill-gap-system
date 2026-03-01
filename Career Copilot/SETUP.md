# 🚀 SETUP & USAGE GUIDE

## Quick Start

### 1. Install All Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 2. Set Up LLM API Key (Optional but Recommended)

**For OpenAI GPT:**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=sk-your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here
```

**For Google Gemini:**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY=your-api-key-here
```

### 3. Run the System

**Option A: Web Interface (Recommended)**

Terminal 1 - Start Backend API:
```bash
python api_server.py
```
Runs at: http://localhost:8000

Terminal 2 - Start UI:
```bash
streamlit run streamlit_app.py
```
Opens at: http://localhost:8501

**Option B: Command Line**
```bash
python app.py
```

---

## 🧪 Testing

### Test Abbreviation Matching
```bash
python test_abbreviations.py
```

### Test RAG Engine
```bash
python -c "from rag_engine import get_rag_engine; rag = get_rag_engine(); print(rag.get_context_for_skill('machine learning'))"
```

###Test API Health
Visit: http://localhost:8000/api/health

---

## 📊 API Endpoints

### POST /api/analyze
Analyze resume vs job description

**Request:**
- `resume`: PDF file
-`job_description`: Text

**Response:**
```json
{
  "match_score": 82.9,
  "matched_skills": [...],
  "missing_skills": [...],
  "career_advice": {...}
}
```

### POST /api/skill-context
Get learning resources for a skill

### GET /api/health
System health check

---

## 🛠️ Troubleshooting

### No LLM API key
System works without API key but won't generate career advice.

### FAISS installation error
```bash
pip install faiss-cpu --force-reinstall
```

### spaCy model not found
```bash
python -m spacy download en_core_web_sm --force
```

---

## 📁 Project Structure

```
├── api_server.py              # FastAPI backend
├── streamlit_app.py            # Streamlit UI
├── app.py                      # CLI interface
├── rag_engine.py               # FAISS RAG system
├── llm_client.py               # GPT/Gemini integration
├── knowledge_base.py           # Career advice database
├── abbreviation_matcher.py     # Abbreviation handling
└── requirements.txt            # Dependencies
```

---

## ✅ Verification Checklist

- [ ] All dependencies installed
- [ ] spaCy model downloaded
- [ ] NLTK data downloaded
- [ ] LLM API key configured (optional)
- [ ] API server starts without errors
- [ ] Streamlit UI loads successfully
- [ ] Test abbreviation matching passes

---

**🎉 You're ready to use Career Copilot!**
