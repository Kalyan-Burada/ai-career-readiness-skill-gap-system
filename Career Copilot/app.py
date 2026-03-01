import os
from resume_parser import extract_resume_text
from text_cleaner import clean_text
from text_utils import split_into_sentences
from embedding_module import generate_embeddings
from similarity_engine import compute_similarity_matrix, classify_gaps
from phrase_extracter import extract_candidate_phrases

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESUME_FILE = os.path.join(BASE_DIR, "ai-product-manager-resume-example.pdf")
JD_FILE     = os.path.join(BASE_DIR, "jd.txt")

# 1.Resume Processing

raw_resume = extract_resume_text(RESUME_FILE)

clean_resume_sentences = [
    clean_text(s)
    for s in split_into_sentences(raw_resume)
    if len(s.strip()) > 3
]

resume_skills = extract_candidate_phrases(clean_resume_sentences)


# 2.JD Processing

with open(JD_FILE, "r", encoding="utf-8") as f:
    raw_jd = f.read()

clean_jd_sentences = [
    clean_text(s)
    for s in split_into_sentences(raw_jd)
    if len(s.strip()) > 0
]

jd_skills = extract_candidate_phrases(clean_jd_sentences)

if not jd_skills or not resume_skills:
    print("[ERROR] No skills extracted. Verify input files and extraction logic.")
    exit()

# 3.Embedding Generation

resume_embeddings = generate_embeddings(resume_skills)
jd_embeddings     = generate_embeddings(jd_skills)

# 4.Similarity Computation

similarity_matrix = compute_similarity_matrix(jd_embeddings, resume_embeddings)

# 5.Skill Gap Classification

matched, missing = classify_gaps(
    jd_skills,
    resume_skills,
    similarity_matrix
)

# 6.Final Output

W = 62
print("=" * W)
print("         CAREER COPILOT  —  SKILL MATCH ANALYSIS")
print("=" * W)
print(f"  Resume : {RESUME_FILE}")
print(f"  JD     : {JD_FILE}")
print("-" * W)
print(f"  JD skills detected     : {len(jd_skills)}")
print(f"  Resume skills detected : {len(resume_skills)}")
print("-" * W)

print(f"\n  MATCHED  ({len(matched)}/{len(jd_skills)}):")
for skill in sorted(matched):
    print(f"    [+] {skill}")

print(f"\n  MISSING  ({len(missing)}/{len(jd_skills)}):")
for skill in sorted(missing):
    print(f"    [-] {skill}")

match_pct = (len(matched) / len(jd_skills) * 100) if jd_skills else 0
print()
print("-" * W)
print(f"  Overall Match Score : {match_pct:.1f}%")
print("=" * W)