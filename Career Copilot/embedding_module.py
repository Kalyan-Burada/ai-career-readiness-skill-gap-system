from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_list):
    return model.encode(text_list, normalize_embeddings=True)
