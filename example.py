from pipeline.sentiment_model import SentimentPipeline
from preprocess.text_cleaner import read_and_split_document

sentiment_model = SentimentPipeline()

text = "😢"

from langid import classify as langid_classify
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException


def detect_language(text):
    text = text.strip()
    langid_lang, langid_conf = langid_classify(text)

    try:
        langdetect_langs = detect_langs(text)
        top_langdetect = langdetect_langs[0]
        langdetect_lang = top_langdetect.lang
        langdetect_prob = top_langdetect.prob
    except LangDetectException:
        langdetect_lang = None
        langdetect_prob = 0.0

    # If both agree and are confident, use it
    if langid_lang == langdetect_lang and langid_conf > 0.9 and langdetect_prob > 0.9:
        return langid_lang

    # If one is more confident, use that
    if langid_conf >= 0.95:
        return langid_lang
    if langdetect_prob >= 0.95:
        return langdetect_lang

    # Prefer langdetect on medium texts
    if len(text) > 20 and langdetect_lang:
        return langdetect_lang

    # Otherwise fallback to langid
    return langid_lang


language = detect_language(text)[:2]
sentence = read_and_split_document(text)
seniments = sentiment_model.analyze_sentiment(sentence, language)
#print(f"Sentiment: {label}, Scores: {scores}, Lang : {language}")
print(seniments)
print(f"Lang Detected : {language}")
