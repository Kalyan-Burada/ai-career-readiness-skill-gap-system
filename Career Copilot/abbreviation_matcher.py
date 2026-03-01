import re


def is_abbreviation(text):
    """
    Detect if a phrase looks like an abbreviation.
    
    Examples:
    - "AI", "KPI", "NLP" → True
    - "CI/CD", "B2B", "A/B testing" → True
    - "machine learning", "python" → False
    """
    # Remove common phrase words to get the core term
    words = text.lower().split()
    
    # Single uppercase word or letters
    if len(words) == 1:
        word = words[0]
        # 2-5 uppercase letters, possibly with slashes/dots
        if re.match(r'^[a-z]{2,5}(/[a-z]{2,5})?$', word):
            return True
    
    # Check for patterns like "a/b testing" (keep the a/b part)
    if re.search(r'\b[a-z]/[a-z]\b', text.lower()):
        return True
    
    return False


def extract_initials(phrase):
    """
    Extract initials from a phrase.
    
    Examples:
    - "key performance indicators" → "kpi"
    - "artificial intelligence" → "ai"
    - "continuous integration continuous deployment" → "cicd"
    - "return on investment" → "roi"
    """
    # Common stop words, but keep short prepositions for 3-letter abbreviations
    # (e.g., "on" is needed for ROI)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'at', 'in'}
    
    words = phrase.lower().split()
    
    # Try with aggressive filtering first (for longer phrases)
    if len(words) > 3:
        filtered = [w for w in words if w not in stop_words]
        if filtered:
            initials_filtered = ''.join(w[0] for w in filtered if w and w[0].isalpha())
            if len(initials_filtered) >= 2:
                return initials_filtered
    
    # Fallback: use all words (for phrases like "return on investment")
    initials_all = ''.join(w[0] for w in words if w and w[0].isalpha())
    return initials_all


def matches_initials(abbr_phrase, full_phrase):
    """
    Check if an abbreviation matches a full phrase by initials.
    
    Returns:
    - True if initials match
    - False otherwise
    
    Examples:
    - ("kpi", "key performance indicators") → True
    - ("ai", "artificial intelligence") → True
    - ("ci/cd", "continuous integration continuous deployment") → True
    - ("roi", "return on investment") → True
    - ("ml", "machine learning") → True
    """
    # Quick check: if one phrase contains the other (handles "a/b testing" vs "a/b testing methodologies")
    abbr_lower = abbr_phrase.lower().strip()
    full_lower = full_phrase.lower().strip()
    
    if abbr_lower in full_lower or full_lower in abbr_lower:
        return True
    
    # Extract the abbreviation part (remove trailing words like "testing", "management")
    abbr_words = abbr_phrase.lower().split()
    
    # Check each word for abbreviation pattern
    for word in abbr_words:
        # Remove slashes and dots
        clean_abbr = word.replace('/', '').replace('.', '')
        
        if len(clean_abbr) >= 2 and clean_abbr.isalpha():
            full_initials = extract_initials(full_phrase)
            
            # Direct match
            if clean_abbr == full_initials:
                return True
            
            # Check if abbreviation is a subset of initials (for compound terms)
            if clean_abbr in full_initials or full_initials in clean_abbr:
                return True
    
    return False


def get_abbreviation_boost(jd_skill, resume_skill):
    """
    Calculate a boost score if one is an abbreviation of the other.
    
    Returns:
    - 1.0 if clear abbreviation match
    - 0.0 otherwise
    """
    # Check both directions (JD could have abbreviation or full form)
    if is_abbreviation(jd_skill) and matches_initials(jd_skill, resume_skill):
        return 1.0
    
    if is_abbreviation(resume_skill) and matches_initials(resume_skill, jd_skill):
        return 1.0
    
    return 0.0
