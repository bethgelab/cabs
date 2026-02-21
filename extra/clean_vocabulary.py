"""
Clean vocabulary_descriptions.json:
1. Remove "caucasian women" (racial concept)
2. Remove plural duplicates (keep singular), except glasses/glass
3. Remove underscore duplicates (keep space version)
4. Remove capitalization duplicates (keep lowercase/normal)
"""

import json
import re
from collections import defaultdict

INPUT_PATH = "/weka/bethge/bkr536/cabs/data/vocabulary_descriptions.json"
OUTPUT_PATH = "/weka/bethge/bkr536/cabs/data/vocabulary_descriptions.json"

with open(INPUT_PATH) as f:
    data = json.load(f)

# Build concept name list
all_concepts = {}
for i, entry in enumerate(data):
    for key in entry:
        all_concepts[key] = i

concept_names = list(all_concepts.keys())

# Collect all names to remove
to_remove = set()

# ============================================================
# 1. Only remove "caucasian women"
# ============================================================
racial_remove = {"caucasian women"}
to_remove.update(racial_remove)

# ============================================================
# 2. Plural duplicates (keep singular)
# ============================================================
PLURAL_EXCEPTIONS = {"glasses"}  # glasses != glass

remaining = {n for n in concept_names if n not in to_remove}
lower_to_names = defaultdict(list)
for n in remaining:
    lower_to_names[n.lower()].append(n)

plural_remove = {}
for name in sorted(remaining):
    if name in PLURAL_EXCEPTIONS:
        continue
    lower = name.lower()
    if lower.endswith("s") and len(lower) > 2:
        singular_candidate = lower[:-1]
        singular_candidate_es = lower[:-2] if lower.endswith("es") else None
        singular_candidate_ies = lower[:-3] + "y" if lower.endswith("ies") else None

        for sc in [singular_candidate, singular_candidate_es, singular_candidate_ies]:
            if sc and sc in lower_to_names:
                singular_actual = lower_to_names[sc][0]
                if name != singular_actual and name not in plural_remove:
                    plural_remove[name] = singular_actual

to_remove.update(plural_remove.keys())

# ============================================================
# 3. Underscore vs space duplicates (keep space version)
# ============================================================
remaining2 = {n for n in concept_names if n not in to_remove}
lower_to_names2 = defaultdict(list)
for n in remaining2:
    lower_to_names2[n.lower()].append(n)

underscore_remove = {}
for name in sorted(remaining2):
    if "_" in name:
        space_lower = name.replace("_", " ").lower()
        if space_lower in lower_to_names2:
            kept = lower_to_names2[space_lower][0]
            if kept != name:
                underscore_remove[name] = kept

to_remove.update(underscore_remove.keys())

# ============================================================
# 4. Capitalization duplicates (keep lowercase/normal)
# ============================================================
remaining3 = {n for n in concept_names if n not in to_remove}
lower_to_names3 = defaultdict(list)
for n in remaining3:
    lower_to_names3[n.lower()].append(n)

capitalization_remove = {}
for lower_key, names in lower_to_names3.items():
    if len(names) <= 1:
        continue
    def sort_key(n):
        if n == n.lower():
            return (0, len(n), n)
        elif n == n.title():
            return (2, len(n), n)
        else:
            return (1, len(n), n)

    sorted_names = sorted(names, key=sort_key)
    keeper = sorted_names[0]
    for other in sorted_names[1:]:
        capitalization_remove[other] = keeper

to_remove.update(capitalization_remove.keys())

# ============================================================
# Apply removals
# ============================================================
print(f"Removing {len(to_remove)} concepts...")
print(f"  1. Racial:          {len(racial_remove)}")
print(f"  2. Plurals:         {len(plural_remove)}")
print(f"  3. Underscores:     {len(underscore_remove)}")
print(f"  4. Capitalization:  {len(capitalization_remove)}")

new_data = []
for entry in data:
    keys = list(entry.keys())
    if not keys:
        continue
    key = keys[0]
    if key not in to_remove:
        new_data.append(entry)

print(f"\nBefore: {len(data)}")
print(f"After:  {len(new_data)}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(new_data, f, indent=4)

print(f"Saved to {OUTPUT_PATH}")
