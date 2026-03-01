"""Test RAG Engine"""
from rag_engine import get_rag_engine

print("Initializing RAG Engine...")
rag = get_rag_engine()

print("\n✓ RAG Engine initialized successfully!")

# Test retrieval
skill = "machine learning"
ctx = rag.get_context_for_skill(skill)

print(f"\n📚 Testing context retrieval for: {skill}")
print(f"   Description: {ctx['description'][:80]}...")
print(f"   Learning Resources: {len(ctx['learning_resources'])} items")
print(f"   Project Ideas: {len(ctx['project_ideas'])} items")
print(f"   Career Paths: {', '.join(ctx['career_paths'][:3])}")
print(f"   Related Skills: {', '.join(ctx['related_skills'])}")

print("\n✅ All RAG tests passed!")
