import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# Entity labels that correspond to skills / tools / technologies / orgs.
# This is NOT a skill list — it's NER-category filtering (pure NLP).
_SKILL_ENT_LABELS = {
    'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW',
    'NORP', 'FAC', 'EVENT',
}

# Prepositions that introduce the "real skill" after a frame noun
# ("experience IN X", "understanding OF X", "familiarity WITH X").
_FRAME_PREPS = {'in', 'of', 'with'}

# Verbs that head responsibility bullet points in JDs.
# The direct object (dobj) of these verbs is the real skill/task.
_ACTION_VERBS = {
    'lead', 'manage', 'conduct', 'analyze', 'analyse', 'monitor',
    'optimize', 'optimise', 'develop', 'design', 'build', 'create',
    'implement', 'drive', 'coordinate', 'collaborate', 'define',
    'deliver', 'oversee', 'execute', 'ensure', 'review', 'evaluate',
    'maintain', 'support', 'improve', 'work', 'use', 'leverage',
}


def extract_candidate_phrases(text_list):
    """
    Extract skill / domain phrases using spaCy dependency parsing,
    NER, verb-object extraction, and POS tags.
    Input text should preserve original casing so that spaCy can
    detect proper nouns and named entities accurately.

    ➜ No predefined skill dictionaries — purely NLP-driven.
    """
    phrases = set()

    for sentence in text_list:
        doc = nlp(sentence)

        # ── 1. Named entities (tool names, companies, tech) ──────────
        for ent in doc.ents:
            if ent.label_ not in _SKILL_ENT_LABELS:
                continue
            phrase = _normalize(ent.text)
            if _is_valid_phrase(phrase):
                phrases.add(phrase)

        # ── 2. Verb-object pairs for JD responsibility bullet points ─
        #    e.g. "Monitor KPIs including revenue growth" → "kpis"
        #    e.g. "Optimize supply chain processes" → "supply chain processes"
        for token in doc:
            if token.pos_ == 'VERB' and token.lemma_.lower() in _ACTION_VERBS:
                for child in token.children:
                    if child.dep_ == 'dobj':
                        # Expand the dobj to its full noun phrase
                        dobj_span = _expand_noun(child, doc)
                        phrase = _normalize(dobj_span)
                        if _is_valid_phrase(phrase):
                            phrases.add(phrase)
                        # Also pick up prepositional objects attached to dobj
                        for gc in child.children:
                            if gc.dep_ == 'prep':
                                for pobj in gc.children:
                                    if pobj.dep_ == 'pobj':
                                        pobj_span = _expand_noun(pobj, doc)
                                        phrase = _normalize(pobj_span)
                                        if _is_valid_phrase(phrase):
                                            phrases.add(phrase)

        # ── 3. Noun chunks with dependency-based filtering ───────────
        for chunk in doc.noun_chunks:
            root = chunk.root

            # Skip sentence subjects ("The ideal candidate …", "We …")
            if root.dep_ in ('nsubj', 'nsubjpass'):
                continue

            # Skip frame nouns whose prep child is in/of/with.
            # The actual skill appears as pobj and is its own chunk.
            if any(ch.dep_ == 'prep' and ch.text.lower() in _FRAME_PREPS
                   for ch in root.children):
                continue

            # Skip conjuncts of frame nouns ("… and optimization of …")
            if root.dep_ == 'conj':
                head = root.head
                if any(ch.dep_ == 'prep' and ch.text.lower() in _FRAME_PREPS
                       for ch in head.children):
                    continue

            # ── Build cleaned token list ─────────────────────────────
            tokens = list(chunk)

            # Strip leading DET / PRON (but protect "A/B" style compounds)
            while tokens and tokens[0].pos_ in ('DET', 'PRON'):
                if len(tokens) > 1 and tokens[1].text in ('/', '-'):
                    break
                tokens = tokens[1:]

            # Strip trailing punctuation
            while tokens and tokens[-1].pos_ == 'PUNCT':
                tokens = tokens[:-1]
            if not tokens:
                continue

            phrase = _normalize(' '.join(t.text for t in tokens))
            pos_tags = [t.pos_ for t in tokens]

            if _is_valid_phrase(phrase, pos_tags):
         phrases.add(phrase)

    # 4. Deduplicate (remove subsets of longer phrases)
    phrases = _deduplicate(phrases)

    return sorted(phrases)

#  Helpers
def _expand_noun(token, doc):
    """
    Expand a noun token to include its left-side compound/amod modifiers,
    giving a fuller phrase (e.g. 'chain' → 'supply chain processes').
    """
    start = token.i
    end = token.i + 1
    # Walk left to include compound/amod modifiers
    for left in token.lefts:
        if left.dep_ in ('compound', 'amod', 'nmod', 'poss'):
            start = min(start, left.i)
    # Walk right to include compound right-children
    for right in token.rights:
        if right.dep_ in ('compound',):
            end = max(end, right.i + 1)
    return ' '.join(t.text for t in doc[start:end])


def _normalize(text):
    """Lowercase, strip, collapse whitespace around hyphens / slashes."""
    text = text.lower().strip().strip(',').strip()
    text = re.sub(r'\s*([/-])\s*', r'\1', text)
    return re.sub(r'\s+', ' ', text)


# Past-participle words that are only meaningful as the suffix of a
# hyphenated compound adjective (e.g. "ai-DRIVEN", "data-DRIVEN",
# "cloud-BASED", "ai-POWERED").  A phrase starting with one of these
# alone is a broken fragment — reject it.
_COMPOUND_SUFFIX_FRAGMENTS = {
    'driven', 'powered', 'based', 'focused', 'enabled', 'oriented',
    'led', 'first', 'facing', 'ready', 'aware',
}


def _is_valid_phrase(phrase, pos_tags=None):
    """
    Accept meaningful skill / domain terms; reject noise.
    Uses POS tags from spaCy — no hardcoded skill dictionaries.
    """
    if not phrase or len(phrase) < 2:
        return False

    words = phrase.split()

    # Phrase starts with a compound-adjective fragment word → reject
    # e.g. "driven digital products" (tail of "ai-driven digital products")
    if words[0] in _COMPOUND_SUFFIX_FRAGMENTS:
        return False

    # Entirely stopwords → reject
    if all(w in stop_words for w in words):
        return False

    # Single-word: strict gate for precision 
    if len(words) == 1:
        word = words[0]
        if word in stop_words or len(word) < 2:
            return False
        # Contains non-alpha chars → likely technical (a/b, kpi, etc.)
        if re.search(r'[^a-z]', word):
            return True
        # Proper noun (tool / technology name detected by spaCy)
        if pos_tags and pos_tags[0] == 'PROPN':
            return True
        # Common-noun alone is too vague → reject for precision
        return False

    # Multi-word
    non_stop = [w for w in words if w not in stop_words]
    if not non_stop:
        return False
    # Must contain at least one NOUN or PROPN
    if pos_tags and not any(p in ('NOUN', 'PROPN') for p in pos_tags):
        return False
    return True


def _deduplicate(phrases):
    """
    Remove phrases whose *unique* content words are an exact subset of a
    longer kept phrase AND they share the same head/root word.

    FIX: Old logic used plain set-subset, which incorrectly removed
    'product backlog' as a subset of 'product roadmap' (both share
    the content word 'product').  New logic requires ALL non-stop
    content words to appear verbatim in the longer phrase's text,
    not just as a mathematical set intersection.
    """
    by_length = sorted(phrases, key=lambda p: len(p.split()), reverse=True)
    kept = []
    for p in by_length:
        p_words = [w for w in p.split() if w not in stop_words]
        if not p_words:
            continue
        is_redundant = False
        for longer in kept:
            # All non-stop words of `p` must appear IN ORDER within `longer`
            if _is_subsequence(p_words, longer.split()):
                is_redundant = True
                break
        if not is_redundant:
            kept.append(p)
    return set(kept)


def _is_subsequence(short_words, long_words):
    """
    Return True only if every word in short_words appears in long_words
    as a contiguous sub-sequence (not just a mathematical subset).
    This prevents 'product backlog' being flagged as subset of
    'product roadmap' even though 'backlog' ≠ 'roadmap'.
    """
    if not short_words:
        return False
    n, m = len(short_words), len(long_words)
    for i in range(m - n + 1):
        if long_words[i:i + n] == short_words:
            return True
    return False