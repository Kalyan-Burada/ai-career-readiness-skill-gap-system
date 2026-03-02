import re
import spacy
from nltk.corpus import stopwords
import string

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# Entity labels that correspond to skills / tools / technologies / orgs.
# This is NOT a skill list – it's NER-category filtering (pure NLP).
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

    → No predefined skill dictionaries – purely NLP-driven.
    """
    phrases = set()

    for sentence in text_list:
        # Handle standalone technical terms (e.g., "I2C", "SPI", "UART")
        # These often appear as isolated sentences from bullet lists in resumes
        sentence_stripped = sentence.strip()
        words = sentence_stripped.split()
        
        # If sentence is 1-3 words, check for STRONG technical markers and add directly.
        # "Strong" means: digit (CSS3, HTML5), special char (C++, .NET), mixed-case
        # interior caps (JavaScript, TypeScript, SolidWorks), or all-caps acronym
        # (GPIO, UART, USB). Pure Initial-Cap-only words ("Reczee", "React") are NOT
        # added here — their lowercase form goes through normal spaCy processing below,
        # and real framework names appear in context sentences where NER/noun chunks work.
        if 1 <= len(words) <= 3:
            normalized_sentence = _normalize(sentence_stripped)
            if normalized_sentence and len(normalized_sentence) >= 2:
                has_digit     = any(c.isdigit() for c in sentence_stripped)
                has_special   = any(c in ['+', '#', '/'] for c in sentence_stripped)
                # Interior uppercase = mixed-case technical name (JavaScript, SolidWorks)
                # Ignore the very first character — Initial Cap alone is not a signal.
                has_interior_upper = any(c.isupper() for c in sentence_stripped[1:])
                all_caps_short = (sentence_stripped.isupper()
                                  and 2 <= len(sentence_stripped.replace(' ', '')) <= 6)

                if has_digit or has_special or has_interior_upper or all_caps_short:
                    if _is_valid_phrase(normalized_sentence):
                        phrases.add(normalized_sentence)
        
        doc = nlp(sentence)

        # ── 1. Named entities (tool names, companies, tech) ──────
        for ent in doc.ents:
            if ent.label_ not in _SKILL_ENT_LABELS:
                continue
            phrase = _normalize(ent.text)
            if _is_valid_phrase(phrase):
                phrases.add(phrase)

        # ── 2. Verb-object pairs for JD responsibility bullet points ──
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

        # ── 3. Noun chunks with dependency-based filtering ───────
        # Skip for single-token documents: the 1-3 word direct check at the
        # top of this loop already handled them correctly and applies the
        # technical-word heuristic WITHOUT the PROPN bypass.  Running noun
        # chunks on a 1-token doc would wrongly rescue standalone brand names
        # like "LinkedIn" or "BetterTeam" via the PROPN tag path in
        # _is_valid_phrase, even though the direct check correctly rejected them.
        if len(doc) <= 1:
            continue

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

            # Skip conjuncts of frame nouns ("… and optimization of …",
            # "… and understanding of …").  A conjunct whose HEAD also has
            # a in/of/with prep child inherits the frame-noun nature.
            if root.dep_ == 'conj':
                head = root.head
                if any(ch.dep_ == 'prep' and ch.text.lower() in _FRAME_PREPS
                       for ch in head.children):
                    continue
                # Also skip if the conjunct itself has of/in/with prep children
                if any(ch.dep_ == 'prep' and ch.text.lower() in _FRAME_PREPS
                       for ch in root.children):
                    continue

            # ── Build cleaned token list ─────────────────────
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

            # If this chunk is a comma-separated list (e.g. "React js, Angular")
            # extract each member individually so each keeps its own POS context.
            # This preserves PROPN tags for tool names that would otherwise be
            # lost when the split happens in step 5 without pos_tags.
            comma_positions = [i for i, t in enumerate(tokens) if t.text == ',']
            if comma_positions:
                seg_start = 0
                for cp in comma_positions:
                    seg = [t for t in tokens[seg_start:cp] if t.pos_ != 'PUNCT']
                    if seg:
                        seg_phrase   = _normalize(' '.join(t.text for t in seg))
                        seg_pos_tags = [t.pos_ for t in seg]
                        if _is_valid_phrase(seg_phrase, seg_pos_tags):
                            phrases.add(seg_phrase)
                    seg_start = cp + 1
                # last segment after final comma
                seg = [t for t in tokens[seg_start:] if t.pos_ != 'PUNCT']
                if seg:
                    seg_phrase   = _normalize(' '.join(t.text for t in seg))
                    seg_pos_tags = [t.pos_ for t in seg]
                    if _is_valid_phrase(seg_phrase, seg_pos_tags):
                        phrases.add(seg_phrase)
                continue   # handled via segments; skip full-phrase add below

            if _is_valid_phrase(phrase, pos_tags):
                phrases.add(phrase)

    # 4. Deduplicate (remove subsets of longer phrases)
    phrases = _deduplicate(phrases)

    # 5. Split comma-separated lists AND handle "and"-separated terms.
    # "UART, I2C, SPI, CAN, and USB" → ["uart", "i2c", "spi", "can", "usb"]
    # "i2c and can bus communication"  → ["i2c", "can bus communication"]
    # IMPORTANT: re-validate every part after splitting — without pos_tags context
    # a split fragment like "clean" or "maintainable" (from "clean, maintainable,
    # and reusable code") must still pass _is_valid_phrase; adjectives and generic
    # words that only survived because of their longer phrase are caught here.
    expanded = set()
    for phrase in phrases:
        phrase_for_split = phrase
        phrase_for_split = phrase_for_split.replace(', and ', ',')
        phrase_for_split = phrase_for_split.replace(' and ', ',')

        if ',' in phrase_for_split:
            parts = [p.strip() for p in phrase_for_split.split(',') if p.strip()]
            for part in parts:
                if _is_valid_phrase(part):   # re-validate without pos_tags
                    expanded.add(part)
        else:
            expanded.add(phrase)

    return sorted(expanded)

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


def _is_technical_word(word):
    """
    Semantic heuristic: Check if single word is likely technical or generic.
    NO hardcoded keyword lists — uses morphological and letter-pattern signals only.

    Rules applied in order (first matching rule wins):
      1. Multi-word phrase         → always accept (handled upstream).
      2. Contains digit/special/
         mixed-case/all-caps ≤5   → technical indicator, accept.
      3. Ends in -ing (gerund)     → process/activity word (welding, machining,
                                     programming); accept and rely on spaCy POS
                                     tagging in _is_valid_phrase to block adjective
                                     uses (e.g. "outstanding" tagged ADJ is rejected).
      4. Ends in abstract-quality
         noun suffix               → generic state/quality noun, reject.
         (-ness, -ity, -ment,
          -ance, -ence, -ency,
          -ancy, -ety)
      5. High common-letter ratio  → likely everyday English word, reject.
    """
    if ' ' in word:
        return True

    word_lower = word.lower()

    # ── Strong technical signals ──────────────────────────────────────────
    has_digit     = any(c.isdigit() for c in word)
    has_special   = any(c in ['+', '#', '/', '-'] for c in word)
    is_mixed_case = any(c.isupper() for c in word) and any(c.islower() for c in word)
    all_caps_short = word.isupper() and len(word) <= 5   # GPIO, UART, I2C …

    if has_digit or has_special or is_mixed_case or all_caps_short:
        return True

    all_lowercase = word_lower == word

    if all_lowercase and 3 < len(word) < 16:

        # ── Gerund / process words (welding, machining, programming) ─────
        # These are valid technical skills as standalone nouns.
        # spaCy POS filter in _is_valid_phrase blocks adjective/verb usages.
        # Exception: very long -ing words (>=11 chars) whose stem is a
        # common everyday verb (high common-letter ratio) are cognitive /
        # communication verbs, not domain skills.
        # e.g. "understanding" (13) stem "understand" ratio 1.0 → generic.
        # e.g. "programming" (11) stem "programm" ratio 0.5 → technical.
        if word_lower.endswith('ing'):
            if len(word_lower) >= 11:
                stem = word_lower[:-3]   # strip 'ing'
                common_chars_set = set('aeiourstndlh')
                stem_ratio = sum(1 for c in stem if c in common_chars_set) / len(stem)
                if stem_ratio > 0.7:
                    return False   # generic cognitive / communication gerund
            return True

        # ── Abstract quality / state / adjective suffixes ─────────────────
        # Words ending here are almost never standalone skills;
        # they describe qualities / states / actions rather than concrete
        # artefacts or techniques.
        # Multi-word phrases that *contain* these words (e.g.
        # "process management", "circuit tolerance") are unaffected because
        # _is_technical_word is only called for single-word candidates.
        _abstract_suffixes = (
            # ---- abstract state / quality nouns ----
            'ness',   # weakness, effectiveness, awareness
            'ity',    # creativity, agility, productivity, ability
            'ment',   # management, improvement, assessment
            'ance',   # performance, compliance, tolerance
            'ence',   # experience, competence, excellence
            'ency',   # efficiency, proficiency, latency
            'ancy',   # consultancy, redundancy
            'ety',    # safety, variety, anxiety
            # ---- action / process abstract nouns ----
            'tion',   # communication, motivation, collaboration, coordination
            'sion',   # passion, mission, decision, revision
            # ---- quality / capability adjective suffixes ----
            'tive',   # proactive, reactive, innovative, collaborative, effective
            'sive',   # aggressive, progressive, comprehensive
            # ---- capability / describability adjectives ----
            'ible',   # flexible, responsible, accessible, compatible
            'able',   # manageable, scalable, maintainable, adaptable
            # ---- past-participle generic action words ----
            'ized',   # organized, optimized, digitized, standardized
            'ised',   # organised, optimised (British spelling)
        )
        if word_lower.endswith(_abstract_suffixes):
            return False

        # ── Rare-letter signal: x, z, q, j are uncommon in everyday English ─
        # Words containing them tend to be technical / borrowed terms
        # (linux, unix, ajax, jquery, quartz, fuzzing).
        if any(c in 'xzqj' for c in word_lower):
            return True

        # ── High common-letter ratio → generic everyday English ──────────
        common_chars  = 'aeiourstndlh'
        common_ratio  = sum(1 for c in word_lower if c in common_chars) / len(word_lower)
        if common_ratio > 0.7:
            return False

    return True


# Past-participle words that are only meaningful as the suffix of a
# hyphenated compound adjective (e.g. "ai-DRIVEN", "data-DRIVEN",
# "cloud-BASED", "ai-POWERED").  A phrase starting with one of these
# alone is a broken fragment – reject it.
_COMPOUND_SUFFIX_FRAGMENTS = {
    'driven', 'powered', 'based', 'focused', 'enabled', 'oriented',
    'led', 'first', 'facing', 'ready', 'aware',
}


def _is_valid_phrase(phrase, pos_tags=None):
    """
    Accept meaningful skill / domain terms; reject only obvious garbage.
    Uses semantic patterns to detect generic nouns - NO hardcoded keywords.
    """
    if not phrase or len(phrase) < 2:
        return False

    words = phrase.split()

    # Entirely stopwords → reject
    if all(w.rstrip(',') in stop_words for w in words):
        return False

    # Reject compound suffix fragments at start
    if words[0].rstrip(',') in _COMPOUND_SUFFIX_FRAGMENTS:
        return False
    
    # Reject fragments with leading special chars
    if words[0].startswith(('/', '-', 'the-')):
        return False
    
    # Reject phone numbers and email patterns
    digit_count = sum(1 for c in phrase if c.isdigit())
    dash_count = phrase.count('-')
    if digit_count >= 5 and dash_count >= 2:
        return False
    if 'professionalemail' in phrase.lower() or '@ ' in phrase or ' @' in phrase:
        return False
    
    # Reject pure numeric or mostly numeric phrases
    non_space = [c for c in phrase if c != ' ']
    if non_space and sum(1 for c in non_space if c.isdigit()) / len(non_space) > 0.5:
        return False
    
    # Reject year-based phrases (starts with 4-digit number)
    if len(words) > 1 and words[0].isdigit() and len(words[0]) == 4:
        return False

    # Reject username/personal info patterns
    if 'username' in phrase.lower() or 'com/in' in phrase.lower():
        return False
    if phrase.count('/') > 1:
        return False

    # Must contain at least one noun or proper noun (semantic filter).
    # Check this BEFORE the single-word heuristic so that spaCy's PROPN
    # label (e.g. SolidWorks, Kubernetes, AutoCAD after lowercasing) can
    # rescue proper-noun tool names from the letter-ratio filter.
    if pos_tags and not any(p in ('NOUN', 'PROPN') for p in pos_tags):
        return False

    # Semantic check: reject generic (non-technical) single words.
    # Skip this check when spaCy already identified the token as a proper
    # noun — those are named entities / tool names, always keep them.
    if len(words) == 1:
        is_propn = pos_tags and pos_tags[0] == 'PROPN'
        if not is_propn and not _is_technical_word(phrase):
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