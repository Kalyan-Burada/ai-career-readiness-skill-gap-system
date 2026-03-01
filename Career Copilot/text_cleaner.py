import re

def clean_text(text):
    # NOTE: do NOT lowercase here — case is needed for accurate
    # POS tagging (PROPN detection) and NER in spaCy.

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters (preserve letters, digits, commas, hyphens, slashes)
    text = re.sub(r"[^a-zA-Z0-9\s,'\-/]", ' ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
