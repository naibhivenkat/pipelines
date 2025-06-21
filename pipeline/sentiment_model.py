from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline.languages import LANGUAGE_MODELS


class SentimentPipeline:

    def __init__(self):
        self.loaded_models = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, lang):
        if lang not in LANGUAGE_MODELS:
            raise ValueError(f"Language '{lang}' not supported.")

        if lang not in self.loaded_models:
            model_name = LANGUAGE_MODELS[lang]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      use_fast=False,
                                                      force_download=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name)
            model.to(self.device)
            model.eval()
            self.loaded_models[lang] = (tokenizer, model)

        return self.loaded_models[lang]

    def get_label_map(self, lang):
        if lang == "en":
            return {0: "NEG", 1: "NEG", 2: "NEU", 3: "POS", 4: "POS"}
        elif lang in {"fr", "ko"}:
            return {0: "NEG", 1: "POS"}
        elif lang in {"zh", "hi", "ta", "bn"}:
            return {0: "NEG", 1: "NEU", 2: "POS"}
        else:
            return {0: "NEG", 1: "POS"}

    def classify_sentiment(self, text, tokenizer, model, label_map):
        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        max_idx = int(np.argmax(probs))
        sentiment = label_map.get(max_idx, "UNK")
        return sentiment, float(probs[max_idx]), probs.tolist()

    def analyze_sentiment(self, sentences, lang="en"):
        tokenizer, model = self.load_model(lang)
        label_map = self.get_label_map(lang)

        sentence_results = []
        for item in sentences:
            sentiment, confidence, _ = self.classify_sentiment(
                item['sentence'], tokenizer, model, label_map)
            sentence_results.append({
                "start_offset": item["start_offset"],
                "end_offset": item["end_offset"],
                "sentiment": sentiment,
                "score": round(confidence, 3),
            })

        # Document-level sentiment
        # full_text = " ".join([read_and_split_document(item.get("cleaned", item["sentence"])) for item in sentences])
        full_text = " ".join(
            [item.get("cleaned", item["sentence"]) for item in sentences])
        sentiment, confidence, _ = self.classify_sentiment(
            full_text, tokenizer, model, label_map)
        document_result = {
            "start_offset": sentences[0]["start_offset"],
            "end_offset": sentences[-1]["end_offset"],
            "sentiment": sentiment,
            "score": round(confidence, 3),
        }

        return {
            "metadata": {
                "version": "1.0",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "language": lang.upper(),
            },
            "sections": sentence_results,
            "sentiment": document_result
        }
