import re


def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u263a-\U0001f645]', '', text)
    text = text.lower()
    return text
