"""Translation support using deep-translator."""

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False


class Translator:
    """Translates text using Google Translate."""

    def __init__(self, target_lang="en"):
        self.target_lang = target_lang
        self._translators = {}  # Cache by (source_lang, target_lang)
        if not TRANSLATION_AVAILABLE:
            print("[WARN] deep-translator not available - translation disabled")
            print("       Install with: pip install deep-translator")

    def _get_translator(self, source_lang):
        """Get or create a cached GoogleTranslator for the given source language."""
        key = (source_lang, self.target_lang)
        if key not in self._translators:
            self._translators[key] = GoogleTranslator(source=source_lang, target=self.target_lang)
        return self._translators[key]

    def translate(self, text, source_lang):
        """Translate text to target language. Returns None if same language or on failure."""
        if not TRANSLATION_AVAILABLE:
            return None
        if source_lang == self.target_lang:
            return None
        try:
            translator = self._get_translator(source_lang)
            result = translator.translate(text)
            return result if result else None
        except Exception:
            return None
