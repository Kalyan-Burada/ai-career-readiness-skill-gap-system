from abbreviation_matcher import is_abbreviation, matches_initials, get_abbreviation_boost

# Test cases
test_cases = [
    ("kpi", "key performance indicators", True),
    ("ai", "artificial intelligence", True),
    ("ml", "machine learning", True),
    ("ci/cd", "continuous integration continuous deployment", True),
    ("roi", "return on investment", True),
    ("nlp", "natural language processing", True),
    ("a/b testing", "a/b testing methodologies", True),
    ("python", "java", False),
    ("product", "engineering", False),
]

print("=" * 70)
print("ABBREVIATION MATCHING TEST")
print("=" * 70)

for abbr, full, expected in test_cases:
    is_abbr = is_abbreviation(abbr)
    match = matches_initials(abbr, full)
    boost = get_abbreviation_boost(abbr, full)
    
    status = "✓" if match == expected else "✗"
    
    print(f"\n{status} '{abbr}' vs '{full}'")
    print(f"   Is abbreviation: {is_abbr}")
    print(f"   Initials match: {match}")
    print(f"   Boost score: {boost}")
    print(f"   Expected: {expected}")

print("\n" + "=" * 70)
