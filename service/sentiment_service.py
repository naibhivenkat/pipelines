from pipeline.sentiment_model import SentimentPipeline
from preprocess.text_cleaner import read_and_split_document
from langid import classify as langid_classify
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

sentiment_model = SentimentPipeline()


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

    if langid_lang == langdetect_lang and langid_conf > 0.9 and langdetect_prob > 0.9:
        return langid_lang
    if langid_conf >= 0.95:
        return langid_lang
    if langdetect_prob >= 0.95:
        return langdetect_lang
    if len(text) > 20 and langdetect_lang:
        return langdetect_lang
    return langid_lang


def analyze_document(text: str) -> dict:
    language = detect_language(text)[:2]
    sentences = read_and_split_document(text)
    sentiments = sentiment_model.analyze_sentiment(sentences, language)
    return {
        "language": language,
        "sentences": sentences,
        "sentiments": sentiments
    }
