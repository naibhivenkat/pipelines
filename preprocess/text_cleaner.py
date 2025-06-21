import re
import emoji
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.resolve().parent
emoticons = BASE_DIR / "emoticons.tsv"


def load_emoticon_dict(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    return dict(zip(df["emoticon"], df["meaning"]))


def normalize_emoticons(text, emoticon_dict):
    for emoticon in emoticon_dict:
        alt_emoticon = emoticon.replace("-", "‑")  # non-breaking hyphen
        text = text.replace(alt_emoticon, emoticon)
    return text


def replace_emoticons(text, emoticon_dict):
    for emoticon, meaning in emoticon_dict.items():
        escaped = re.escape(emoticon)
        text = re.sub(escaped, f" {meaning} ", text)
    return text


def extract_sentences_with_offsets(text):
    pattern = re.compile(r'[^.!?。！？]*[.!?。！？]', flags=re.UNICODE)
    matches = list(pattern.finditer(text))
    sentences = []

    last_end = 0
    for match in matches:
        sentence = match.group().strip()
        start = match.start()
        end = match.end()
        sentences.append({
            "sentence": sentence,
            "start_offset": start,
            "end_offset": end
        })
        last_end = end

    # Handle any trailing text that wasn't matched (no final punctuation)
    if last_end < len(text):
        remainder = text[last_end:].strip()
        if remainder:
            sentences.append({
                "sentence": remainder,
                "start_offset": last_end,
                "end_offset": len(text)
            })

    return sentences


def read_and_split_document(text):
    emoticon_dict = load_emoticon_dict(emoticons)
    sentence_offsets = extract_sentences_with_offsets(text)
    cleaned_sentences = []

    for item in sentence_offsets:
        sentence = item['sentence']
        sentence = normalize_emoticons(sentence, emoticon_dict)
        sentence = replace_emoticons(sentence, emoticon_dict)
        sentence = emoji.demojize(sentence, language='en')
        sentence = re.sub(r':([a-zA-Z0-9_]+):', r'\1', sentence)
        sentence = re.sub(r'[^a-zA-Z_\s]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        cleaned_sentences.append({
            "sentence": sentence,
            "start_offset": item['start_offset'],
            "end_offset": item['end_offset']
        })

    return cleaned_sentences
