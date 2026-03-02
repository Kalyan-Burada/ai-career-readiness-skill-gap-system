from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from abbreviation_matcher import get_abbreviation_boost


def compute_similarity_matrix(jd_embeddings, resume_embeddings):
    return cosine_similarity(jd_embeddings, resume_embeddings)


def _normalize_skill(text):
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)


def _tokenize_skill(text):
    # Generic tokenizer only; no hardcoded domain keywords.
    return [token for token in re.findall(r'[a-z0-9+#/-]+', _normalize_skill(text)) if len(token) >= 2]


def _has_lexical_evidence(jd_skill, resume_skill):
    """
    Non-hardcoded lexical check to prevent semantic-only false positives.
    - Exact normalized string -> True
    - Abbreviation equivalence -> True
    - Single-token JD skill requires exact token presence in resume skill
    - Multi-token JD skill requires meaningful token overlap
    """
    jd_norm = _normalize_skill(jd_skill)
    resume_norm = _normalize_skill(resume_skill)

    if jd_norm == resume_norm:
        return True

    if get_abbreviation_boost(jd_skill, resume_skill) > 0:
        return True

    jd_tokens = _tokenize_skill(jd_skill)
    resume_tokens = set(_tokenize_skill(resume_skill))

    if not jd_tokens or not resume_tokens:
        return False

    # Example: uart, spi, mips must appear exactly in resume-side tokens
    if len(jd_tokens) == 1:
        return jd_tokens[0] in resume_tokens

    # Multi-token overlap ratio, without any keyword lists.
    overlap = sum(1 for token in jd_tokens if token in resume_tokens)
    return (overlap / len(jd_tokens)) >= 0.5


def classify_gaps(
    jd_skills,
    resume_skills,
    similarity_matrix,
    threshold=0.92,
):
    """
    Classify each JD skill as MATCHED or MISSING against the resume.

        Strategy:
        1) Exact/abbreviation/lexical evidence against extracted resume skills.
        2) Cosine fallback only at high threshold and only with lexical evidence.

        This avoids semantic-only matches that are not actually present in resume text.
    """
    matched = []
    missing = []

    for i, jd_skill in enumerate(jd_skills):
        # First pass: require lexical evidence or abbreviation with any resume skill.
        direct_match = False
        for resume_skill in resume_skills:
            if _has_lexical_evidence(jd_skill, resume_skill):
                matched.append(jd_skill)
                direct_match = True
                break

        if direct_match:
            continue

        sims = similarity_matrix[i]

        # Cosine fallback with lexical guard
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_resume_skill = resume_skills[best_idx]

        if best_sim >= threshold and _has_lexical_evidence(jd_skill, best_resume_skill):
            matched.append(jd_skill)
            continue

        missing.append(jd_skill)

    return matched, missing