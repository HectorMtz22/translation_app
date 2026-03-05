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

    def translate(self, text, source_lang, context=None):
        """Translate text to target language. Returns None if same language or on failure.

        If context is provided (list of recent lines), it is prepended to
        improve translation quality and only the actual translation is returned.
        """
        if not TRANSLATION_AVAILABLE:
            return None
        if source_lang == self.target_lang:
            return None
        try:
            translator = self._get_translator(source_lang)
            if context:
                # Prepend context lines separated by newlines, add a marker
                # so we can split the result reliably
                marker = "|||"
                block = "\n".join(context) + f"\n{marker}\n" + text
                result = translator.translate(block)
                if result and marker in result:
                    return result.split(marker, 1)[1].strip() or None
                # Fallback: marker got translated/lost, just translate without context
            result = translator.translate(text)
            return result if result else None
        except Exception:
            return None
