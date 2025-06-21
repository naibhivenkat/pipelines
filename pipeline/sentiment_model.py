# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from preprocess.text_cleaner import clean_text
# from languages import LANGUAGE_MODELS
#
#
# class SentimentPipeline:
#     def __init__(self):
#         self.loaded_models = {}
#
#     def load_model(self, lang):
#         if lang not in LANGUAGE_MODELS:
#             raise ValueError(f"Language '{lang}' not supported.")
#         if lang not in self.loaded_models:
#             model_name = LANGUAGE_MODELS[lang]
#             tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#             model = AutoModelForSequenceClassification.from_pretrained(model_name)
#
#             model.eval()
#             self.loaded_models[lang] = (tokenizer, model)
#         return self.loaded_models[lang]
#
#     def __call__(self, text: str, lang: str):
#         tokenizer, model = self.load_model(lang)
#         cleaned = clean_text(text)
#         inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         scores = torch.nn.functional.softmax(outputs.logits, dim=1)
#         label = torch.argmax(scores).item()
#         label_map = {0: "NEG", 1: "NEG", 2: "NEU", 3: "POS", 4: "POS"}
#
#         return label_map[label], scores.tolist()[0]
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess.text_cleaner import clean_text
from languages import LANGUAGE_MODELS


class SentimentPipeline:
    def __init__(self):
        self.loaded_models = {}

    def load_model(self, lang):
        if lang not in LANGUAGE_MODELS:
            raise ValueError(f"Language '{lang}' not supported.")
        if lang not in self.loaded_models:
            model_name = LANGUAGE_MODELS[lang]
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            self.loaded_models[lang] = (tokenizer, model)
        return self.loaded_models[lang]

    def __call__(self, text: str, lang: str):
        tokenizer, model = self.load_model(lang)
        cleaned = clean_text(text)
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_index = torch.argmax(scores).item()
        confidence = float(scores[0][label_index])
        confidence = round(confidence, 3)

        # Map label index to 3-class label
        label_map = {0: "NEG", 1: "NEG", 2: "NEU", 3: "POS", 4: "POS"}
        label = label_map.get(label_index, "UNK")

        return label, confidence
