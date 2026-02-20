"""English translation support using deep-translator."""

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False


class Translator:
    """Translates text to English using Google Translate."""

    def __init__(self):
        if not TRANSLATION_AVAILABLE:
            print("[WARN] deep-translator not available - translation disabled")
            print("       Install with: pip install deep-translator")

    def translate_to_english(self, text, source_lang):
        """Translate text to English. Returns original text if already English or on failure."""
        if not TRANSLATION_AVAILABLE:
            return None
        if source_lang == "en":
            return None
        try:
            result = GoogleTranslator(source=source_lang, target="en").translate(text)
            return result if result else None
        except Exception:
            return None
