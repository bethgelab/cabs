import json
import re
from pathlib import Path
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# NLTK setup
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

# Load sentence transformer model
print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient


# ---------------------------------------------------------
# Embedding-Based Safety Check
# ---------------------------------------------------------

# Define unsafe concept examples for embedding comparison
UNSAFE_RACE_EXAMPLES = [
    "white person", "black person", "Asian person", "Latino person",
    "white man", "black woman", "Asian man", "Hispanic woman",
    "white people", "black people", "Asian people", "Latino people",
    "Caucasian person", "African American person", "Native American person",
    "white male", "black female", "Asian male", "Latino female",
    "white individual", "black individual", "racial group", "ethnic person"
]

UNSAFE_NSFW_EXAMPLES = [
    "nude person", "naked body", "sexual intercourse", "pornography",
    "genitals", "explicit nudity", "sex act", "masturbation",
    "sexual organs", "erotic nudity", "pornographic content",
    "sexual penetration", "oral sex", "genital contact",
    "nude sexual", "naked sexual", "explicit sex", "condom usage",
    "sexual protection device", "contraceptive device for intercourse"
]

# Safe examples to help distinguish
SAFE_EXAMPLES = [
    "Asian elephant", "African lion", "European robin", "Indian Ocean",
    "black cat", "white house", "brown bear", "red fox",
    "United States", "South Africa", "Asian cuisine", "Latin America",
    "black color", "white paint", "ancient Egypt", "Medieval Europe",
    "woman standing", "man walking", "people gathering", "female athlete",
    "male dancer", "fashion model", "body part", "human body", "person running",
    "white background", "black background", "white shirt", "black dress",
    "bikini top", "swimming top", "clothing top", "fashion top",
    "acting performance", "theater act", "circus act", "magic act",
    "ship mast", "flagpole mast", "sailing mast", "boat mast",
    "organ music", "church organ", "body organ", "organ system",
    "nude mouse", "hairless mouse", "laboratory mouse", "genetic mouse"
]

def create_safety_classifier():
    """Create embeddings for unsafe concept categories"""
    print("Creating safety classifier embeddings...")
    
    race_embeddings = model.encode(UNSAFE_RACE_EXAMPLES)
    nsfw_embeddings = model.encode(UNSAFE_NSFW_EXAMPLES)
    safe_embeddings = model.encode(SAFE_EXAMPLES)
    
    # Compute centroids
    race_centroid = np.mean(race_embeddings, axis=0)
    nsfw_centroid = np.mean(nsfw_embeddings, axis=0)
    safe_centroid = np.mean(safe_embeddings, axis=0)
    
    return {
        'race_centroid': race_centroid,
        'nsfw_centroid': nsfw_centroid,
        'safe_centroid': safe_centroid,
        'race_embeddings': race_embeddings,
        'nsfw_embeddings': nsfw_embeddings,
        'safe_embeddings': safe_embeddings
    }


def check_concept_safety(concept, classifier, race_threshold=0.70, nsfw_threshold=0.65):
    """
    Check if a concept is unsafe using embedding similarity.
    
    Args:
        concept: The concept string to check
        classifier: Dict containing embeddings and centroids
        race_threshold: Similarity threshold for race classification (higher = stricter)
        nsfw_threshold: Similarity threshold for NSFW classification (higher = stricter)
    
    Returns:
        "SAFE", "UNSAFE_RACE", or "UNSAFE_NSFW"
    """
    # Encode the concept
    concept_embedding = model.encode([concept])[0]
    
    # Compute max similarity to individual examples for precision
    max_race_sim = np.max(cosine_similarity(
        [concept_embedding], 
        classifier['race_embeddings']
    ))
    
    max_nsfw_sim = np.max(cosine_similarity(
        [concept_embedding], 
        classifier['nsfw_embeddings']
    ))
    
    max_safe_sim = np.max(cosine_similarity(
        [concept_embedding], 
        classifier['safe_embeddings']
    ))
    
    # Decision logic: unsafe if similar to unsafe examples AND significantly more similar to unsafe than safe
    # Require unsafe similarity to be higher than both threshold AND safe similarity
    if max_race_sim > race_threshold and max_race_sim > max_safe_sim + 0.05:
        return "UNSAFE_RACE"
    
    if max_nsfw_sim > nsfw_threshold and max_nsfw_sim > max_safe_sim + 0.05:
        return "UNSAFE_NSFW"
    
    return "SAFE"


def check_concepts_batch(concepts, classifier):
    """Check safety for a batch of concepts"""
    results = {}
    
    for i, concept in enumerate(concepts):
        if i % 100 == 0:
            print(f"Checking concept {i+1}/{len(concepts)}...")
        
        results[concept] = check_concept_safety(concept, classifier)
    
    return results


# ---------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------

def normalize_spacing(s):
    return re.sub(r"\s+", " ", s.strip())


def normalize_key(key):
    """Remove underscores and convert to lowercase"""
    return key.replace("_", " ").lower()


def morphological_normalize(key):
    tokens = key.split()
    return " ".join(lemmatizer.lemmatize(t, pos="n") for t in tokens)


def get_synset_key(word):
    """Deprecated - not used anymore, kept for compatibility"""
    syns = wordnet.synsets(word.replace(" ", "_"))
    return syns[0].name() if syns else None


def are_words_spelling_variants(word1, word2):
    """
    Check if two words are spelling variants or abbreviations.
    Examples: "hair drier" vs "hair dryer", "rv" vs "recreational vehicle"
    """
    from difflib import SequenceMatcher
    
    w1_lower = word1.lower()
    w2_lower = word2.lower()
    
    # Handle abbreviations
    w1_tokens = w1_lower.split()
    w2_tokens = w2_lower.split()
    
    # Check if one is an abbreviation of the other
    if len(w1_tokens) == 1 and len(w2_tokens) > 1:
        abbrev = ''.join([t[0] for t in w2_tokens])
        if w1_tokens[0] == abbrev:
            return True
    if len(w2_tokens) == 1 and len(w1_tokens) > 1:
        abbrev = ''.join([t[0] for t in w1_tokens])
        if w2_tokens[0] == abbrev:
            return True
    
    # Check for common spelling variants (British vs American, etc.)
    # Only merge if words differ by 1-2 characters and share same root
    if len(w1_lower) > 4 and len(w2_lower) > 4:
        # Calculate edit distance for spelling variants
        similarity = SequenceMatcher(None, w1_lower, w2_lower).ratio()
        if similarity > 0.92:  # Very high similarity (spelling variant)
            return True
    
    return False


def are_concepts_duplicates(word1, word2, embedding_threshold=0.95):
    """
    Check if two concepts are essentially duplicates using embeddings.
    
    ONLY merge if:
    1. Spelling variants (hair drier vs hair dryer)
    2. Exact synonyms with >0.95 embedding similarity
    
    DO NOT merge:
    - Different species (crayfish vs lobster)
    - Different objects (palace vs castle, cd vs common dandelion)
    - Related but distinct concepts (lakeside vs lakeshore)
    
    Args:
        word1, word2: Concepts to compare
        embedding_threshold: Cosine similarity threshold (0.95 = extremely similar)
    
    Returns:
        True if concepts should be merged, False otherwise
    """
    # First check if they're spelling variants or abbreviations
    if are_words_spelling_variants(word1, word2):
        return True
    
    # Use embeddings to check semantic similarity
    # Only merge if embeddings are EXTREMELY similar (>0.95)
    embeddings = model.encode([word1, word2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Very strict threshold: only merge near-identical concepts
    # "cd" vs "common dandelion" will have low similarity (~0.3)
    # "hair drier" vs "hair dryer" will have high similarity (~0.98)
    # "crayfish" vs "spiny lobster" will have medium similarity (~0.7-0.8)
    return similarity > embedding_threshold


# ---------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------

def clean_concepts(input_json_path, output_json_path):

    data = json.loads(Path(input_json_path).read_text())

    # Flatten JSON list-of-dicts into a normal dict
    concept_dict = {}
    for entry in data:
        for k, v in entry.items():
            concept_dict[k] = v

    print(f"Initial number of concepts: {len(concept_dict)}")

    # =====================================================
    # 0. Normalize keys: remove underscores and lowercase
    # =====================================================
    normalized_dict = {}
    removed_normalization = 0
    
    for key, val in concept_dict.items():
        norm_key = normalize_key(key)
        if norm_key in normalized_dict:
            print(f"KEY NORMALIZATION: merged '{key}' → kept '{norm_key}'")
            removed_normalization += 1
        else:
            normalized_dict[norm_key] = val
    
    print(f"Key normalization merges: {removed_normalization}")

    # =====================================================
    # 1. Syntactic normalization
    # =====================================================
    synt_map = {}
    removed_syntax = 0

    for key, val in normalized_dict.items():
        norm = normalize_spacing(key)
        if norm in synt_map:
            print(f"SYNTAX MERGE: merged '{key}' → kept '{norm}'")
            removed_syntax += 1
        else:
            synt_map[norm] = val

    print(f"Syntactical redundancies removed: {removed_syntax}")

    # =====================================================
    # 2. Morphological normalization
    # =====================================================
    morph_map = {}
    removed_morph = 0

    for key, val in synt_map.items():
        morph = morphological_normalize(key)
        if morph in morph_map:
            print(f"MORPH MERGE: merged '{key}' → kept '{morph}'")
            removed_morph += 1
        else:
            morph_map[morph] = val

    print(f"Morphological redundancies removed: {removed_morph}")

    # =====================================================
    # 3. Strict semantic merging via WordNet synsets
    # =====================================================
    synset_map = {}
    removed_semantic = 0

    for key, val in morph_map.items():
        # Find if there's an existing key that's a duplicate
        merged = False
        for existing_key in list(synset_map.keys()):
            if are_concepts_duplicates(key, existing_key, embedding_threshold=0.95):
                print(f"SEMANTIC MERGE: merged '{key}' → kept '{existing_key}'")
                removed_semantic += 1
                merged = True
                break
        
        if not merged:
            synset_map[key] = val

    print(f"Semantic redundancies removed: {removed_semantic}")

    # =====================================================
    # 4. Embedding-based unsafe concept removal
    # =====================================================
    print("\n" + "="*60)
    print("Running embedding-based safety check...")
    print("="*60 + "\n")
    
    # Create classifier
    classifier = create_safety_classifier()
    
    # Check all concepts
    safety_results = check_concepts_batch(list(synset_map.keys()), classifier)
    
    final_dict = {}
    removed_unsafe = 0
    removed_by_type = {"UNSAFE_RACE": 0, "UNSAFE_NSFW": 0}

    for key, val in synset_map.items():
        safety_status = safety_results.get(key, "SAFE")
        
        if safety_status != "SAFE":
            print(f"UNSAFE CONCEPT REMOVAL ({safety_status}): '{key}'")
            removed_unsafe += 1
            removed_by_type[safety_status] = removed_by_type.get(safety_status, 0) + 1
        else:
            final_dict[key] = val

    print(f"\nUnsafe concepts removed: {removed_unsafe}")
    print(f"  - Race-related: {removed_by_type.get('UNSAFE_RACE', 0)}")
    print(f"  - NSFW: {removed_by_type.get('UNSAFE_NSFW', 0)}")
    print(f"Final number of concepts: {len(final_dict)}")

    # Save final JSON
    out_list = [{k: v} for k, v in final_dict.items()]
    Path(output_json_path).write_text(json.dumps(out_list, indent=2))

    print(f"\nSaved cleaned JSON to: {output_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean and deduplicate concept vocabulary")
    parser.add_argument("--input", type=str, required=True, help="path to input concept JSON")
    parser.add_argument("--output", type=str, required=True, help="path to output cleaned JSON")
    args = parser.parse_args()
    clean_concepts(args.input, args.output)