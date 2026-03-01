from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from abbreviation_matcher import get_abbreviation_boost


def compute_similarity_matrix(jd_embeddings, resume_embeddings):
    return cosine_similarity(jd_embeddings, resume_embeddings)


def classify_gaps(
    jd_skills,
    resume_skills,
    similarity_matrix,
    threshold=0.45,          # Primary match threshold
    soft_threshold=0.35,     # Secondary semantic fallback threshold (tuned for paraphrase matching)
    top_k=5,                 # Number of top resume skills to consider in fallback
):
    """
    Classify each JD skill as MATCHED or MISSING against the resume.

    Strategy:
    - Primary  : best cosine similarity ≥ threshold → MATCHED
    - Secondary: average of top-k cosine similarities ≥ soft_threshold → MATCHED
      (handles cases where one phrase partially matches several resume phrases,
       e.g. 'artificial intelligence concepts' matching 'ai', 'machine learning', etc.)
    - Abbreviation: if one skill is abbreviation of another → MATCHED
      (handles KPI ↔ key performance indicators, AI ↔ artificial intelligence, etc.)
    """
    matched = []
    missing = []

    for i, jd_skill in enumerate(jd_skills):
        sims = similarity_matrix[i]

        # Primary check: best single match
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        if best_sim >= threshold:
            matched.append(jd_skill)
            continue

        # Secondary check: top-k average (semantic spread)
        top_k_indices = np.argsort(sims)[-top_k:]
        avg_top_k = np.mean(sims[top_k_indices])

        if avg_top_k >= soft_threshold:
            matched.append(jd_skill)
            continue

        # Tertiary check: abbreviation matching
        # Check if JD skill is an abbreviation that matches any resume skill
        found_abbr_match = False
        for resume_skill in resume_skills:
            if get_abbreviation_boost(jd_skill, resume_skill) > 0:
                matched.append(jd_skill)
                found_abbr_match = True
                break
        
        if not found_abbr_match:
            missing.append(jd_skill)

    return matched, missing