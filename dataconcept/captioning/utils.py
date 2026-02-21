import pickle
import re


def count_files_pkl(pkl_path: str) -> int:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return len(data)
    raise ValueError("The pickle file does not contain a dictionary.")


def trim_caption(caption: str) -> str:
    """trim to the last complete sentence if the caption was cut off mid-sentence."""
    if caption.count('.') > 1 and not caption.strip().endswith('.'):
        sentences = re.findall(r'[^.]*\.', caption)
        return ''.join(sentences).strip()
    return caption.strip()


EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F'
    r'\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF'
    r'\U0001F700-\U0001F77F'
    r'\U0001F780-\U0001F7FF'
    r'\U0001F800-\U0001F8FF'
    r'\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F'
    r'\U0001FA70-\U0001FAFF'
    r'\U00002702-\U000027B0'
    r'\U000024C2-\U0001F251]+'
)

PUNCTUATION = r"\(\[\{\?\.,!|\-/<>~`@#_=;:\]\}\)"


def preprocess_caption(caption):
    """clean up alt-text: remove emojis, fix spacing, normalize punctuation."""
    if caption is None:
        return ''

    caption = EMOJI_PATTERN.sub('', caption)

    # collapse double-spaced text (common OCR artifact)
    caption = ' '.join(''.join(word) for word in caption.split('  '))

    caption = re.sub(rf'\s+([{PUNCTUATION}])', r'\1', caption)
    caption = re.sub(rf'([{PUNCTUATION}])\1+', r'\1', caption)
    caption = re.sub(rf'([{PUNCTUATION}])([^\s{PUNCTUATION}])', r'\1 \2', caption)

    return caption.strip()


def clean_and_capitalize(description):
    """strip model preamble and leading filler phrases, then capitalize."""
    description = re.sub(r".*?\nassistant\n", "", description, flags=re.DOTALL)
    cleaned = re.sub(r"^(The image\s+\w+\s+(an|the|a)\s?)", "", description, flags=re.IGNORECASE)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def build_prompt(caption, classes, max_caption_len=77):
    """build the recaptioning prompt from alt-text and detected classes."""
    alt = caption[:max_caption_len] if (len(caption) > max_caption_len or '#' not in caption) else caption
    parts = ["Briefly caption the image using relevant details from the alt-text and detected classes."]
    parts.append(f"Alt-text: {alt}.")
    if classes:
        parts.append(f"Classes: {', '.join(map(str, classes))}.")
    return " ".join(parts)
