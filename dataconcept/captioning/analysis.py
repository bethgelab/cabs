"""evaluate whether generated captions mention all detected object classes."""

import json
import re
import os
import logging
import argparse
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import Levenshtein

for pkg in ['wordnet', 'averaged_perceptron_tagger', 'punkt', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# text matching
# ---------------------------------------------------------------------------

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            synonyms.add(name)
            if '_' in name:
                synonyms.add(name.replace('_', ' '))
    return synonyms


def get_word_forms(word, include_synonyms=False):
    """generate morphological variants: lemma, plurals, gerunds, and optionally synonyms."""
    word = word.replace('-', ' ').lower()
    lemmatized = lemmatizer.lemmatize(word)
    forms = {word, lemmatized}

    if include_synonyms:
        for w in [word, lemmatized] + (word.split() if ' ' in word else []):
            forms.update(get_synonyms(w))

    for base in list(forms):
        # gerunds
        forms.add(base + 'ing')
        if base.endswith('e'):
            forms.add(base[:-1] + 'ing')
        # plurals
        if base.endswith('y'):
            forms.add(base[:-1] + 'ies')
        elif base.endswith(('s', 'sh', 'ch', 'x', 'z')):
            forms.add(base + 'es')
        else:
            forms.add(base + 's')

    return forms


def normalize_text(text):
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'\s+-\s+', ' ', text)


def levenshtein_similarity(a, b):
    dist = Levenshtein.distance(a.lower(), b.lower())
    return 1 - dist / max(len(a), len(b))


def find_best_match(target_forms, caption_words, caption_forms):
    for target in target_forms:
        if target in caption_forms:
            return 1.0, target, target

    best_score, best_target, best_word = 0, None, None
    for target in target_forms:
        for word in caption_words:
            score = levenshtein_similarity(target, word)
            if score > best_score:
                best_score, best_target, best_word = score, target, word
    return best_score, best_target, best_word


# ---------------------------------------------------------------------------
# class-in-caption checking
# ---------------------------------------------------------------------------

def check_classes_in_caption(caption, classes, strategy='vanilla', threshold=0.85):
    """check whether each class appears in the caption.

    strategies:
      - vanilla: exact word-form matching
      - synonyms: exact + wordnet synonyms
      - full: synonyms + levenshtein partial matching
    """
    normalized = normalize_text(caption)
    words = word_tokenize(normalized)

    use_syn = strategy in ('synonyms', 'full')
    caption_forms = set()
    for w in words:
        caption_forms.update(get_word_forms(w, include_synonyms=use_syn))

    results = {}
    all_found = True

    for idx, cls in enumerate(classes):
        cls_norm = cls.replace('-', ' ')
        cls_forms = get_word_forms(cls_norm, include_synonyms=use_syn)

        match = {'class': cls_norm, 'idx': idx, 'found': False,
                 'matched_form': None, 'matched_word': None, 'score': 0.0}

        # exact match
        for form in cls_forms:
            if form in normalized or form in caption_forms:
                match.update(found=True, matched_form=form, matched_word=form, score=1.0)
                break

        # fuzzy match (full strategy only)
        if not match['found'] and strategy == 'full':
            score, best, word = find_best_match(cls_forms, words, caption_forms)
            if score >= threshold:
                match.update(found=True, matched_form=best, matched_word=word, score=score)

        results[f"{cls}_{idx}"] = match
        if not match['found']:
            all_found = False

    return all_found, results


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def setup_logging(strategy, log_prefix):
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger(f'caption_analysis_{strategy}')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in [logging.FileHandler(f'{log_prefix}_{strategy}_{ts}.log'),
                    logging.StreamHandler()]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def analyze_captions(data, captions, log_prefix, strategy='vanilla', threshold=0.85):
    """check every image's caption for its required classes and log results."""
    logger = setup_logging(strategy, log_prefix)
    logger.info(f"strategy: {strategy}, threshold: {threshold}")

    results = {}
    total, missing_images, total_cls, missing_cls = 0, 0, 0, 0

    for item in data:
        fname = item['filename']
        if fname not in captions:
            continue

        total += 1
        caption = captions[fname]
        classes = [c.replace('-', ' ') for c in item['classes']]
        total_cls += len(classes)

        logger.info(f"\n{fname} | classes: {classes}")
        logger.info(f"caption: {caption}")

        all_found, class_results = check_classes_in_caption(
            caption, classes, strategy=strategy, threshold=threshold,
        )
        results[fname] = {'caption': caption, 'classes': classes,
                          'all_found': all_found, 'details': class_results}

        has_missing = False
        for details in class_results.values():
            if not details['found']:
                has_missing = True
                missing_cls += 1
                logger.warning(f"  missing: {details['class']} (#{details['idx']})")
                if strategy == 'full' and details['score'] > 0:
                    logger.info(f"  closest: {details['matched_word']} ({details['score']:.3f})")
            else:
                logger.info(f"  found: {details['class']} (#{details['idx']}) as '{details['matched_form']}'")
        if has_missing:
            missing_images += 1

    rate = (total_cls - missing_cls) / total_cls * 100 if total_cls else 0
    logger.info(f"\n=== summary === images: {total}, instances: {total_cls}, "
                f"missing: {missing_cls}, success: {rate:.1f}%")
    return results


def run_all_strategies(data, captions, log_prefix):
    all_results = {}
    for strategy in ('vanilla', 'synonyms', 'full'):
        print(f"\nrunning {strategy} strategy...")
        all_results[strategy] = analyze_captions(data, captions, log_prefix, strategy=strategy)

        r = all_results[strategy]
        n_missing = sum(1 for v in r.values() if not v['all_found'])
        rate = (len(r) - n_missing) / len(r) * 100 if r else 0
        print(f"  success rate: {rate:.1f}%")
    return all_results


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_metadata(folder_path):
    """load class labels from json files in a folder (one json per image)."""
    data = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(folder_path, fname)) as f:
            meta = json.load(f)
        data.append({
            'filename': fname.replace('.json', '.jpg'),
            'classes': meta.get('classes', []),
        })
    return data


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate caption quality by checking class mention rates")
    parser.add_argument("--captions", type=str, required=True, help="path to captions json file")
    parser.add_argument("--metadata", type=str, required=True, help="folder with per-image json metadata")
    parser.add_argument("--log_prefix", type=str, default="logs/captions/analysis", help="prefix for log files")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    with open(args.captions) as f:
        captions = json.load(f)
    data = load_metadata(args.metadata)
    run_all_strategies(data, captions, args.log_prefix)
