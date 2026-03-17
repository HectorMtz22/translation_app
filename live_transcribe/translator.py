"""Translation support using deep-translator."""

import time
from collections import OrderedDict

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

TRANSLATION_CACHE_SIZE = 256


class Translator:
    """Translates text using Google Translate with caching and retry."""

    def __init__(self, target_lang="en"):
        self.target_lang = target_lang
        self._translators = {}  # Cache by (source_lang, target_lang)
        self._cache = OrderedDict()  # LRU cache: (text, source_lang) -> translation
        if not TRANSLATION_AVAILABLE:
            print("[WARN] deep-translator not available - translation disabled")
            print("       Install with: pip install deep-translator")

    def _get_translator(self, source_lang):
        """Get or create a cached GoogleTranslator for the given source language."""
        key = (source_lang, self.target_lang)
        if key not in self._translators:
            self._translators[key] = GoogleTranslator(source=source_lang, target=self.target_lang)
        return self._translators[key]

    def _cache_get(self, text, source_lang):
        """Look up a cached translation."""
        key = (text.strip(), source_lang)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, text, source_lang, result):
        """Store a translation in the cache."""
        key = (text.strip(), source_lang)
        self._cache[key] = result
        if len(self._cache) > TRANSLATION_CACHE_SIZE:
            self._cache.popitem(last=False)

    def translate(self, text, source_lang, context=None):
        """Translate text to target language. Returns None if same language or on failure.

        If context is provided (list of recent lines), it is prepended to
        improve translation quality and only the actual translation is returned.
        Uses LRU cache to avoid re-translating identical text, and retries on failure.
        """
        if not TRANSLATION_AVAILABLE:
            return None
        if source_lang == self.target_lang:
            return None

        # Check cache (only for context-free translations to keep it simple)
        if not context:
            cached = self._cache_get(text, source_lang)
            if cached is not None:
                return cached

        # Retry with exponential backoff (2 attempts)
        for attempt in range(2):
            try:
                translator = self._get_translator(source_lang)
                if context:
                    # Prepend context lines separated by newlines, add a marker
                    # so we can split the result reliably
                    marker = "|||"
                    # Limit context to last 5 lines to avoid exceeding API limits
                    recent_ctx = context[-5:] if len(context) > 5 else context
                    # Extract original text from (original, translation) tuples
                    recent_ctx = [c[0] if isinstance(c, tuple) else c for c in recent_ctx]
                    block = "\n".join(recent_ctx) + f"\n{marker}\n" + text
                    result = translator.translate(block)
                    if result and marker in result:
                        translated = result.split(marker, 1)[1].strip() or None
                        if translated:
                            self._cache_put(text, source_lang, translated)
                        return translated
                    # Fallback: marker got translated/lost, just translate without context
                result = translator.translate(text)
                if result:
                    self._cache_put(text, source_lang, result)
                return result if result else None
            except Exception:
                if attempt == 0:
                    time.sleep(0.3)
                    # Re-create translator in case the connection went stale
                    key = (source_lang, self.target_lang)
                    self._translators.pop(key, None)
                continue
        return None
