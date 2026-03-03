"""English translation support using deep-translator."""

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False


class Translator:
    """Translates text to English using Google Translate."""

    def __init__(self):
        self._translators = {}  # Cache by source_lang
        if not TRANSLATION_AVAILABLE:
            print("[WARN] deep-translator not available - translation disabled")
            print("       Install with: pip install deep-translator")

    def _get_translator(self, source_lang):
        """Get or create a cached GoogleTranslator for the given source language."""
        if source_lang not in self._translators:
            self._translators[source_lang] = GoogleTranslator(source=source_lang, target="en")
        return self._translators[source_lang]

    def translate_to_english(self, text, source_lang):
        """Translate text to English. Returns original text if already English or on failure."""
        if not TRANSLATION_AVAILABLE:
            return None
        if source_lang == "en":
            return None
        try:
            translator = self._get_translator(source_lang)
            result = translator.translate(text)
            return result if result else None
        except Exception:
            return None
