"""
Quick diagnostic: prints raw resume text + extracted resume skills
to see exactly what's in the PDF and what skills get extracted.
"""
from resume_parser import extract_resume_text
from text_cleaner import clean_text
from text_utils import split_into_sentences
from phrase_extracter import extract_candidate_phrases

RESUME_FILE = "ai-product-manager-resume-example.pdf"

raw = extract_resume_text(RESUME_FILE)

print("=" * 70)
print("RAW RESUME TEXT")
print("=" * 70)
print(raw)

sentences = [clean_text(s) for s in split_into_sentences(raw) if len(s.strip()) > 3]

skills = extract_candidate_phrases(sentences)

print("\n" + "=" * 70)
print(f"EXTRACTED RESUME SKILLS ({len(skills)} total)")
print("=" * 70)
for s in skills:
    print(f"  {s}")

# Check the 6 missing JD skills against raw resume text
print("\n" + "=" * 70)
print("MISSING SKILL CHECK — does the raw resume text contain these?")
print("=" * 70)
missing_jd_skills = [
    "cross-functional team leadership",
    "engineering teams",
    "monitor kpis",
    "kpis",
    "optimize supply chain processes",
    "supply chain",
    "related field",
    "strong analytical",
    "analytical and problem-solving",
    "problem-solving skills",
]
raw_lower = raw.lower()
for skill in missing_jd_skills:
    found = skill.lower() in raw_lower
    print(f"  {'FOUND' if found else 'NOT FOUND':12s}  '{skill}'")
