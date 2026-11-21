import re
import unicodedata
from pyvi import ViTokenizer

def tokenize_syllables(text: str) -> list[str]:
    """Chuẩn hóa NFC, chuyển về chữ thường và lấy các cụm chữ hoặc số."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[^\W_]+", text)

def tokenize_words(text: str) -> list[str]:
    """Tách từ tiếng Việt và giữ dấu gạch dưới bên trong từ ghép."""
    text = unicodedata.normalize("NFC", text)
    segmented = ViTokenizer.tokenize(text)
    return re.findall(r"[^\W_]+(?:_[^\W_]+)*", segmented.lower())

STOPWORDS = {"của", "và", "các", "những"}


def preprocess(text, *, unit="syllable", remove_stopwords=False):
    if unit == "syllable":
        tokens = tokenize_syllables(text)
    elif unit == "word":
        tokens = tokenize_words(text)
    else:
        raise ValueError("unit phải là 'syllable' hoặc 'word'")

    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    return tokens
