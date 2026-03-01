"""
RAG (Retrieval-Augmented Generation) Module
Uses FAISS for efficient vector search in knowledge base.
"""

import numpy as np
import faiss
from embedding_module import generate_embeddings
from knowledge_base import get_all_knowledge_texts, get_skill_knowledge


class RAGEngine:
    def __init__(self):
        self.documents = get_all_knowledge_texts()
        self.index = None
        self.doc_embeddings = None
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from knowledge base documents."""
        print("Building RAG knowledge base index...")
        
        # Extract text from documents
        doc_texts = [doc["text"] for doc in self.documents]
        
        # Generate embeddings
        self.doc_embeddings = generate_embeddings(doc_texts)
        
        # Create FAISS index (L2 distance)
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(self.doc_embeddings.astype('float32'))
        
        print(f"✓ Indexed {len(self.documents)} knowledge documents")
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Skill name or query text
            top_k: Number of documents to retrieve
        
        Returns:
            List of (document, distance) tuples
        """
        # Generate query embedding
        query_embedding = generate_embeddings([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Retrieve documents
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": float(dist),
                    "similarity": 1 / (1 + float(dist))  # Convert distance to similarity
                })
        
        return results
    
    def get_context_for_skill(self, skill):
        """
        Get rich context for a missing skill.
        Combines direct lookup + RAG retrieval.
        
        Returns:
            Dictionary with learning resources, projects, career paths
        """
        # Direct lookup
        knowledge = get_skill_knowledge(skill)
        
        # RAG retrieval for additional context
        retrieved = self.retrieve(skill, top_k=2)
        
        # Combine context
        context = {
            "skill": skill,
            "description": knowledge["description"],
            "learning_resources": knowledge["learning_resources"],
            "project_ideas": knowledge["project_ideas"],
            "career_paths": knowledge["career_paths"],
            "estimated_time": knowledge["estimated_time"],
            "related_skills": [r["document"]["skill"] for r in retrieved if r["document"]["skill"] != skill]
        }
        
        return context
    
    def get_context_for_missing_skills(self, missing_skills):
        """Get context for all missing skills."""
        contexts = []
        for skill in missing_skills:
            contexts.append(self.get_context_for_skill(skill))
        return contexts


# Global RAG engine instance (initialized once)
_rag_engine = None

def get_rag_engine():
    """Get or create RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
