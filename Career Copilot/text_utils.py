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

        # Skip pure section header lines (lines ending with ONLY a colon,
        # with no meaningful content before it — e.g. "Responsibilities:")
        stripped = line.rstrip()
        if stripped.endswith(':'):
            # Only skip if the part before ':' is a short header (≤5 words)
            before_colon = stripped[:-1].strip()
            if len(before_colon.split()) <= 5:
                continue

        # For "Label: content" lines where label is a short header (≤5 words),
        # extract only the content part.
        # IMPORTANT: Do NOT split on colons that are inside meaningful content
        # (e.g. "Strong understanding of Artificial Intelligence: concepts").
        # Only split when label looks like a header (title-cased or ≤3 words).
        if ':' in line:
            before, _, after = line.partition(':')
            before = before.strip()
            after  = after.strip()
            # Strip only when: before-colon is a short (≤3 words) header-like prefix
            if after and len(before.split()) <= 3:
                line = after
            # Otherwise keep the full line (colon is inside sentence content)

        # Split on sentence-ending punctuation (period and semicolon)
        parts = re.split(r'[.;]', line)
        for part in parts:
            part = part.strip()
            if len(part) > 2:
                fragments.append(part)

    return fragments