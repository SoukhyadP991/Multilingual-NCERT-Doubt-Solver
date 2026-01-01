from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Reproducible results
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns ISO 639-1 code (e.g., 'en', 'hi', 'ur').
    Default to 'en' on failure.
    """
    try:
        if not text or len(text.strip()) < 3:
            return 'en'
        return detect(text)
    except LangDetectException:
        return 'en'
