import re

def split_into_sentences(text):
    """Split text into meaningful fragments, filtering out section headers."""
    lines = text.split('\n')
    fragments = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Strip leading non-ASCII (emoji prefixes like 🔹)
        line = re.sub(r'^[^\x00-\x7F]+\s*', '', line).strip()
        if not line:
            continue

        # Skip all-caps section headers (SKILLS, WORK EXPERIENCE, etc.)
        if line == line.upper() and len(line.split()) <= 4 and line.replace(' ', '').isalpha():
            continue

        # Skip Title Case structural headers (Job Summary, Key Responsibilities, etc.)
        # Count only alphabetic words — ignore punctuation tokens like "&", "+", "/".
        # This catches "Required Skills & Qualifications" (3 alpha words) as well as
        # plain "Job Summary" (2 words).  NOT CamelCase tools like "SolidWorks".
        _line_words = line.split()
        _sep_tokens  = {'&', '+', '/', '-', '|', 'and', 'or'}
        _alpha_words = [w for w in _line_words if re.match(r'^[A-Z][a-z]+$', w)]
        _other_words = [w for w in _line_words
                        if not re.match(r'^[A-Z][a-z]+$', w) and w not in _sep_tokens]
        if (
            not any(c.isdigit() for c in line)
            and 2 <= len(_alpha_words) <= 4
            and len(_other_words) == 0   # line contains ONLY Initial-Cap words + separators
        ):
            continue

        # Skip single Initial-Cap-only word lines — these are job-board source citations
        # ("Reczee", "Indeed", "Roadmap") or plain nouns, NOT technical skill names.
        # Real tool names that appear here (React, Python) will be extracted from
        # their context in longer sentences anyway.
        if (
            len(_line_words) == 1
            and re.match(r'^[A-Z][a-z]{1,}$', _line_words[0])
            and not any(c.isdigit() for c in line)
        ):
            continue

        # Skip pure section header lines (lines ending with ONLY a colon,
        # with no meaningful content before it — e.g. "Responsibilities:")
        stripped = line.rstrip()
        if stripped.endswith(':'):
            # Only skip if the part before ':' is a short header (≤5 words)
            before_colon = stripped[:-1].strip()
            if len(before_colon.split()) <= 5:
                continue

        # Split on sentence-ending punctuation FIRST.
        # Use '\. ' (period-space) NOT bare '.' to avoid splitting inside
        # technology names like "React.js", "Vue.js", ".NET", "node.js".
        # Semicolons are always sentence separators so split on those freely.
        parts = re.split(r'(?<=\S)\. +|;', line)
        
        for part in parts:
            part = part.strip()
            if len(part) <= 2:
                continue
                
            # For "Label: content" patterns where label is a short header (≤3 words),
            # extract only the content part.
            # IMPORTANT: Do NOT split on colons that are inside meaningful content
            # (e.g. "Strong understanding of Artificial Intelligence: concepts").
            # Only split when label looks like a header (title-cased or ≤3 words).
            if ':' in part:
                before, _, after = part.partition(':')
                before = before.strip()
                after  = after.strip()
                # Strip only when: before-colon is a short header-like prefix.
                # Count only alphabetic words — ignore "&", "+", "/" tokens.
                # This handles "Bridge Design & Tech:" (3 alpha words → strip label)
                # as well as plain "Tools:" (1 word → strip).
                before_alpha_words = [w for w in before.split() if re.match(r'^[A-Za-z]', w)]
                if after and len(before_alpha_words) <= 4:
                    part = after
                # Otherwise keep the full part (colon is inside sentence content)
            
            # Skip fragments that are domain names / URLs.
            # e.g. "arc.dev" → clean_text gives "arc dev" which spaCy
            # extracts as a phrase — avoid this by dropping the raw fragment.
            if re.search(r'\.[a-z]{2,5}$', part.rstrip(), re.IGNORECASE):
                continue

            if len(part) > 2:
                fragments.append(part)

    return fragments