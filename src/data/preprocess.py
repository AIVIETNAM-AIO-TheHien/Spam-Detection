import re
import string
from typing import List

def clean_text(text: str, lower: bool = True, remove_punct: bool = True,
               remove_urls: bool = True, remove_emails: bool = True,
               remove_html: bool = True, remove_numbers: bool = False) -> str:
    """Làm sạch văn bản với nhiều tùy chọn."""
    if not isinstance(text, str):
        text = str(text)
    if remove_html:
        text = re.sub(r'<.*?>', '', text)
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    if remove_emails:
        text = re.sub(r'\S+@\S+', '', text)
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    if lower:
        text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return text.split()

def preprocess_pipeline(text: str) -> List[str]:
    cleaned = clean_text(text)
    return tokenize(cleaned)