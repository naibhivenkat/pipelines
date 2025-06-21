from pipeline.sentiment_model import SentimentPipeline

sentiment_model = SentimentPipeline()

text = "The economic outlook is promising this quarter."
language = "en"
label, scores = sentiment_model(text, language)
print(f"Sentiment: {label}, Scores: {scores}")
