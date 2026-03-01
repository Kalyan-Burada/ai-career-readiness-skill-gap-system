from embedding_module import generate_embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Test if AI and artificial intelligence get matched
jd_skills = ["artificial intelligence concepts"]
resume_skills = ["ai-driven digital products", "machine learning", "data science"]

jd_emb = generate_embeddings(jd_skills)
resume_emb = generate_embeddings(resume_skills)

similarity_matrix = cosine_similarity(jd_emb, resume_emb)

print("JD Skill: 'artificial intelligence concepts'")
print("\nSimilarities with resume skills:")
for i, resume_skill in enumerate(resume_skills):
    score = similarity_matrix[0][i]
    print(f"  {resume_skill:40} → {score:.3f}")

print(f"\nBest match: {resume_skills[similarity_matrix[0].argmax()]}")
print(f"Best score: {similarity_matrix[0].max():.3f}")
print(f"Match threshold: 0.45")
print(f"Result: {'MATCHED' if similarity_matrix[0].max() >= 0.45 else 'MISSING'}")
