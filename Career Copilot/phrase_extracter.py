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

# Generic adjective/descriptive phrases that are NOT real skills.
# These get extracted by NLP but are meaningless as skill gaps.
_NON_SKILL_PHRASES = {
    # Quality / adjective phrases
    'high-quality', 'high quality', 'low-quality', 'low quality',
    'seamless functionality', 'seamless experience', 'seamless integration',
    'reusable code', 'clean code', 'scalable code', 'maintainable code',
    'readable code', 'modular code', 'efficient code', 'production-ready code',
    'technical feasibility', 'technical excellence', 'technical debt',
    'pixel-perfect', 'pixel perfect', 'cross-browser compatibility',

    # Employment / logistics terms
    'full-time', 'full time', 'part-time', 'part time', 'remote',
    'hybrid', 'on-site', 'contract', 'permanent', 'temporary',
    'competitive salary', 'benefits package', 'equal opportunity',

    # Task / deliverable descriptions (NOT skills)
    'new user-facing features', 'user-facing features',
    'new features', 'key features', 'core features',
    'mobile and desktop devices', 'mobile devices', 'desktop devices',
    'multiple platforms', 'various platforms',
    'translate ui/ux design wireframes', 'design wireframes',
    'translate designs', 'implement designs',
    'technical specifications', 'functional requirements',
    'code reviews', 'code review', 'peer reviews',
    'bug fixes', 'bug fixing', 'troubleshooting issues',
    'cross-browser testing', 'manual testing',
    'documentation updates', 'technical documentation',

    # Generic JD filler phrases
    'ideal candidate', 'strong experience', 'good understanding',
    'strong understanding', 'deep understanding', 'solid understanding',
    'related field', 'digital products', 'web applications',
    'similar tools', 'similar analytics tools', 'real-world constraints',
    'various stakeholders', 'key stakeholders', 'team members',
    'junior team members', 'senior team members',
    'best practices', 'industry standards', 'business requirements',
    'technical requirements', 'user engagement', 'user experience',
    'customer feedback', 'performance metrics', 'product metrics',
    'business metrics', 'revenue growth', 'user acquisition',
    'strong analytical', 'preferred qualifications', 'required skills',
    'responsibilities', 'qualifications', 'nice to have',
    'years experience', 'years of experience', 'work experience',
    'bachelor degree', "bachelor's degree", 'master degree', "master's degree",
    'bachelor s', 'bachelors', 'masters', 'bachelor', 'master',
    'bachelor s degree', 'master s degree', 'bachelor of science', 'master of science',
    'phd', 'doctorate', 'associate degree', 'mba',
    'computer science', 'related fields', 'equivalent experience',
    'to present insights', 'present insights', 'to present',
    'to stakeholders', 'to management', 'to leadership',
    'to drive', 'to ensure', 'to support', 'to deliver',
    'to build', 'to create', 'to develop', 'to implement',
    'to maintain', 'to manage', 'to lead', 'to work',
    'fast-paced environment', 'dynamic environment', 'team player',
    'excellent communication', 'communication skills',
    'written and verbal', 'attention to detail', 'detail oriented',
    'self-motivated', 'proactive', 'collaborative environment',
    'proven track record', 'track record', 'strong portfolio',
    'modern web', 'large-scale applications', 'large scale applications',
    'complex applications', 'enterprise applications',
    'startup environment', 'agile environment', 'fast-paced team',
    'product requirements', 'business logic', 'end users',
    'senior developers', 'junior developers', 'other developers',
    'engineering teams', 'development team', 'product team',
    'design team', 'cross-functional teams',

    # JD section headers and structure
    'job summary', 'job description', 'job overview', 'job posting',
    'job details', 'position summary', 'position overview',
    'role summary', 'role overview', 'about the role', 'about us',
    'who we are', 'what we offer', 'what you will do',
    'key responsibilities', 'core responsibilities', 'main responsibilities',
    'primary responsibilities', 'your responsibilities',
    'minimum qualifications', 'preferred qualifications', 'required qualifications',
    'about the company', 'about this role', 'the opportunity',
    'what you bring', 'what we look for', 'your impact',

    # Job platform references
    'indeed jobs', 'indeed', 'glassdoor', 'linkedin jobs',
    'job board', 'job boards', 'career page', 'apply now',
    'remote work options', 'work options', 'flexible work',
    'work arrangements', 'work from home',

    # Generic problem / context descriptions
    'real-world problems', 'real world problems',
    'complex real-world problems', 'complex, real-world problems',
    'diverse sources', 'diverse data sources', 'various sources',
    'production systems', 'production environment', 'production environments',
    'product features', 'key features', 'product roadmap features',
    'related quantitative field', 'quantitative field',
    'relevant field', 'similar field', 'related discipline',
    'day-to-day operations', 'day to day operations',
    'daily operations', 'business operations',
    'actionable insights', 'data-driven insights', 'data driven insights',
    'meaningful insights', 'valuable insights',
    'cross-functional collaboration', 'cross functional collaboration',
    'continuous improvement', 'process improvement',
    'competitive advantage', 'strategic decisions', 'informed decisions',

    # ── Business / Sales / Hospitality non-skill phrases ──
    # Job titles & role descriptions
    'business travel sales manager', 'sales manager', 'account manager',
    'regional sales manager', 'area sales manager', 'national sales manager',
    'hotel general manager', 'front desk manager', 'revenue manager',
    'inside sales representative', 'sales representative', 'sales associate',
    'account executive', 'business development manager',
    'business development representative',

    # Business entity types (not skills)
    'existing high-value corporate clients', 'high-value corporate clients',
    'corporate clients', 'corporate accounts', 'existing corporate accounts',
    'new and existing corporate accounts', 'existing accounts',
    'new corporate accounts', 'key accounts', 'strategic accounts',
    'travel management companies tmcs', 'travel management companies',
    'tmcs', 'travel agencies', 'corporate agencies',
    'new b2b business', 'b2b business', 'b2b clients',
    'new business', 'existing business', 'new clients',

    # Business activities / deliverables (not skills)
    'action plans', 'sales blitzes', 'sales plans',
    'annual rates', 'negotiated rates', 'corporate rates',
    'negotiated corporate rates', 'commercial terms', 'contract terms',
    'corporate room nights', 'room nights', 'room revenue',
    'monthly/quarterly revenue goals', 'revenue goals', 'sales goals',
    'quarterly revenue goals', 'monthly revenue goals', 'sales targets',
    'weekly/monthly sales reports', 'sales reports', 'monthly sales reports',
    'weekly sales reports', 'quarterly reports', 'weekly reports',
    'property site inspections', 'site inspections', 'property tours',
    'industry events', 'networking events', 'trade shows',
    'sales efforts', 'marketing efforts', 'outreach efforts',
    'client relations', 'client relationships', 'customer relationships',
    'account acquisition', 'account retention',
    'competitor activity', 'competitive analysis',

    # Generic business/sales terms that are activities not skills
    'sales strategy', 'business strategy', 'go-to-market strategy',
    'market analysis', 'market trends', 'market research',
    'monthly targets', 'quarterly targets', 'annual targets',
    'revenue growth', 'sales growth', 'business growth',
    'client acquisition', 'lead generation', 'pipeline management',
    'contract negotiation', 'rate negotiation',
    'booking process', 'booking errors', 'travel booking',
    'reimbursement turnaround', 'travel expenses',
    'product knowledge', 'brand awareness',
    'customer engagement', 'customer satisfaction',
    'on-time deliveries', 'on time deliveries',
    'file retrieval times', 'average delivery times',
    'customer wait times', 'wait times',

    # Experience/requirement descriptors (not skills)
    'years of experience', '3-5 years', '5+ years', '3+ years',
    'years experience', 'proven experience', 'strong experience',
    'specifically targeting', 'strong proficiency',
    'excellent negotiation', 'excellent communication',
    'networking skills', 'negotiation skills',
    'excellent negotiation and networking skills',
    'excellent negotiation and networking',
}

# Adjective words that when they are the ONLY non-stop content in a phrase
# make it a quality descriptor, not a skill (e.g. "reusable code", "seamless functionality")
_QUALITY_ADJECTIVES = {
    'reusable', 'seamless', 'scalable', 'maintainable', 'clean',
    'robust', 'efficient', 'reliable', 'secure', 'responsive',
    'interactive', 'intuitive', 'elegant', 'modular', 'flexible',
    'readable', 'testable', 'extensible', 'performant', 'optimized',
    'high-quality', 'production-ready', 'pixel-perfect',
    'new', 'existing', 'modern', 'current', 'latest', 'various',
    'multiple', 'complex', 'simple', 'basic', 'advanced',
    'full-time', 'part-time', 'remote', 'hybrid',
    # Additional context/modifier adjectives that are NOT technical qualifiers
    'diverse', 'key', 'main', 'primary', 'core', 'overall',
    'general', 'specific', 'particular', 'related', 'relevant',
    'similar', 'real-world', 'cross-functional', 'day-to-day',
    'actionable', 'meaningful', 'valuable', 'strategic',
    'competitive', 'continuous', 'informed', 'production',
    # Business/sales descriptors
    'annual', 'monthly', 'weekly', 'quarterly', 'daily',
    'corporate', 'commercial', 'negotiated', 'high-value',
    'existing', 'potential', 'prospective', 'targeted',
    'excellent', 'strong', 'proven', 'effective',
}

# Generic object nouns that are NOT skills by themselves
# (e.g. "code", "features", "functionality" — vague without qualifier)
_GENERIC_OBJECT_NOUNS = {
    'code', 'features', 'functionality', 'components', 'applications',
    'solutions', 'interfaces', 'systems', 'services', 'modules',
    'products', 'platforms', 'tools', 'technologies', 'frameworks',
    'websites', 'pages', 'elements', 'layouts', 'designs',
    'wireframes', 'mockups', 'prototypes', 'specifications',
    'feasibility', 'requirements', 'deliverables', 'milestones',
    'objectives', 'goals', 'outcomes', 'updates', 'issues',
    # Additional generic nouns that are NOT skills by themselves
    'problems', 'sources', 'options', 'field', 'fields',
    'summary', 'responsibilities', 'environment', 'environments',
    'constraints', 'insights', 'operations', 'decisions',
    'advantage', 'improvement', 'collaboration', 'arrangements',
    'jobs', 'work', 'discipline',
    # Business / sales generic nouns (NOT skills)
    'rates', 'terms', 'contracts', 'agreements', 'proposals',
    'reports', 'targets', 'revenue', 'budget', 'accounts',
    'clients', 'customers', 'prospects', 'leads', 'agencies',
    'events', 'meetings', 'inspections', 'tours',
    'nights', 'bookings', 'reservations', 'relationships',
    'efforts', 'activities', 'initiatives', 'strategies',
    'trends', 'analysis', 'research', 'plans',
    'blitzes', 'campaigns', 'presentations', 'pitches',
}

# Generic single words that spaCy may tag as PROPN but are not skills
_NON_SKILL_SINGLE_WORDS = {
    'experience', 'knowledge', 'understanding', 'familiarity',
    'ability', 'skill', 'skills', 'expertise', 'proficiency',
    'requirements', 'responsibilities', 'qualifications',
    'bachelor', 'bachelors', 'master', 'masters', 'phd', 'doctorate', 'mba',
    'present', 'insights', 'stakeholders', 'leadership',
    'team', 'teams', 'role', 'roles', 'candidate', 'company',
    'environment', 'organization', 'department', 'industry',
    'customers', 'clients', 'users', 'stakeholders', 'partners',
    'growth', 'improvement', 'development', 'management',
    'strategy', 'performance', 'quality', 'impact', 'success',
    'opportunities', 'challenges', 'solutions', 'results',
    'process', 'processes', 'standards', 'practices',
    'code', 'features', 'functionality', 'feasibility',
    'components', 'applications', 'platforms', 'tools',
    'interfaces', 'designs', 'wireframes', 'updates',
    'documentation', 'deployment', 'implementation',
    # Additional non-skill single words
    'jobs', 'job', 'summary', 'field', 'fields', 'options',
    'problems', 'sources', 'devices', 'mathematics',
    'indeed', 'key', 'production', 'diverse', 'operations',
    'insights', 'decisions', 'collaboration', 'arrangements',
    'discipline', 'constraints', 'advantage', 'work',
    # Business / sales single words that are NOT skills
    'rates', 'terms', 'contracts', 'revenue', 'budget',
    'accounts', 'clients', 'customers', 'prospects', 'leads',
    'agencies', 'events', 'meetings', 'inspections', 'tours',
    'nights', 'bookings', 'reservations', 'efforts', 'activities',
    'trends', 'analysis', 'research', 'plans', 'blitzes',
    'campaigns', 'pitches', 'targets', 'relationships',
    'networking', 'retention', 'acquisition', 'negotiation',
    'hospitality', 'marketing', 'sales',
}


def _is_valid_phrase(phrase, pos_tags=None):
    """
    Accept meaningful skill / domain terms; reject noise.
    Uses POS tags from spaCy — no hardcoded skill dictionaries.
    Also filters out generic non-skill descriptive phrases,
    quality attributes, and task descriptions.
    """
    if not phrase or len(phrase) < 2:
        return False

    words = phrase.split()

    # Reject known non-skill phrases (exact match)
    if phrase in _NON_SKILL_PHRASES:
        return False

    # Also check with commas stripped (e.g. "complex, real-world problems")
    phrase_no_commas = phrase.replace(',', '').strip()
    phrase_no_commas = re.sub(r'\s+', ' ', phrase_no_commas)
    if phrase_no_commas in _NON_SKILL_PHRASES:
        return False

    # Reject phrases containing commas — usually lists or descriptions, not skills
    # Exception: known tech patterns like "tableau, power bi" with recognized terms
    if ',' in phrase:
        # Check if it looks like a tool/tech list (both parts look technical)
        parts = [p.strip() for p in phrase.split(',') if p.strip()]
        # If any part is more than 3 words, it's a description, not a tool list
        if any(len(p.split()) > 3 for p in parts):
            return False
        # If any part is a known non-skill, reject the whole thing
        if any(p in _NON_SKILL_PHRASES or p in _NON_SKILL_SINGLE_WORDS for p in parts):
            return False

    # Reject education/degree terms
    if re.match(r"^(bachelor|master|phd|doctorate|associate|mba)\b", phrase):
        return False
    
    # Reject infinitive phrases ("to present insights", "to drive growth")
    if phrase.startswith('to ') and len(words) >= 2:
        return False

    # Reject employment type terms as single "skills"
    if phrase.replace('-', '') in {'fulltime', 'parttime', 'remote', 'hybrid', 'onsite', 'contract'}:
        return False

    # Phrase starts with a compound-adjective fragment word → reject
    if words[0].rstrip(',') in _COMPOUND_SUFFIX_FRAGMENTS:
        return False

    # Entirely stopwords → reject
    if all(w.rstrip(',') in stop_words for w in words):
        return False

    # Single-word: strict gate for precision 
    if len(words) == 1:
        word = words[0].rstrip(',')
        if word in stop_words or len(word) < 2:
            return False
        # Reject known non-skill single words
        if word in _NON_SKILL_SINGLE_WORDS:
            return False
        # Reject quality adjectives as standalone
        if word in _QUALITY_ADJECTIVES:
            return False
        # Reject generic object nouns as standalone
        if word in _GENERIC_OBJECT_NOUNS:
            return False
        # Contains non-alpha chars → likely technical (a/b, kpi, etc.)
        # But reject if it's just a hyphenated non-skill like "full-time"
        if re.search(r'[^a-z]', word):
            # Check if it's a hyphenated quality/employment term
            parts = re.split(r'[-/]', word)
            if all(p in _QUALITY_ADJECTIVES or p in _GENERIC_OBJECT_NOUNS
                   or p in _NON_SKILL_SINGLE_WORDS or p in stop_words
                   for p in parts if p):
                return False
            return True
        # Proper noun (tool / technology name detected by spaCy)
        if pos_tags and pos_tags[0] == 'PROPN':
            return True
        # Common-noun alone is too vague → reject for precision
        return False

    # ── Multi-word validation ──
    # Strip commas from words for matching
    clean_words = [w.rstrip(',') for w in words]
    non_stop = [w for w in clean_words if w not in stop_words]
    if not non_stop:
        return False

    # If all meaningful words are in non-skill single words, reject
    if all(w in _NON_SKILL_SINGLE_WORDS for w in non_stop):
        return False

    # Reject "quality-adjective + generic-noun" patterns
    # e.g. "reusable code", "seamless functionality", "new features",
    #       "diverse sources", "production systems", "key responsibilities"
    content_words = [w for w in clean_words if w not in stop_words]
    if len(content_words) >= 2:
        adjectives_in_phrase = [w for w in content_words if w in _QUALITY_ADJECTIVES]
        nouns_in_phrase = [w for w in content_words if w in _GENERIC_OBJECT_NOUNS]
        # If every content word is either a quality adj or a generic noun → reject
        if len(adjectives_in_phrase) + len(nouns_in_phrase) >= len(content_words):
            return False

    # Reject phrases where all content words are non-skill singles
    if all(w in _NON_SKILL_SINGLE_WORDS or w in _QUALITY_ADJECTIVES
           or w in _GENERIC_OBJECT_NOUNS for w in content_words):
        return False

    # Reject phrases that start with a verb (task descriptions like
    # "translate ui/ux design wireframes", "implement new features")
    if pos_tags and pos_tags[0] == 'VERB':
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