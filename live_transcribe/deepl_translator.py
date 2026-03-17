"""Translation support using DeepL API."""

import time
from collections import OrderedDict

try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

import os
from dotenv import load_dotenv

load_dotenv()

TRANSLATION_CACHE_SIZE = 256


class DeepLTranslator:
    """Translates text using DeepL API with caching and retry."""

    # Map short language codes to DeepL source language codes
    LANG_MAP = {
        "ko": "KO",
        "es": "ES",
        "en": "EN",
    }

    # Map short language codes to DeepL target language codes
    TARGET_LANG_MAP = {
        "en": "EN-US",
        "ko": "KO",
        "es": "ES",
    }

    def __init__(self, target_lang="en"):
        self.target_lang = target_lang
        self._cache = OrderedDict()
        if not DEEPL_AVAILABLE:
            print("[WARN] deepl not available - DeepL translation disabled")
            print("       Install with: pip install deepl")
            self.client = None
            return

        api_key = os.environ.get("DEEPL_API_KEY")
        if not api_key:
            print("[WARN] DEEPL_API_KEY not set - DeepL translation disabled")
            print("       Set DEEPL_API_KEY in .env file")
            self.client = None
            return

        self.client = deepl.Translator(api_key)

    def _cache_get(self, text, source_lang):
        key = (text.strip(), source_lang)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, text, source_lang, result):
        key = (text.strip(), source_lang)
        self._cache[key] = result
        if len(self._cache) > TRANSLATION_CACHE_SIZE:
            self._cache.popitem(last=False)

    def translate(self, text, source_lang, context=None):
        """Translate text to target language. Returns None if same language or on failure.

        If context is provided (list of recent lines), it is passed to DeepL's
        native context parameter for better translation quality.
        Uses LRU cache and retries on transient failures.
        """
        if self.client is None:
            return None
        if source_lang == self.target_lang:
            return None

        if not context:
            cached = self._cache_get(text, source_lang)
            if cached is not None:
                return cached

        for attempt in range(2):
            try:
                deepl_lang = self.LANG_MAP.get(source_lang, source_lang.upper())
                deepl_target = self.TARGET_LANG_MAP.get(self.target_lang, self.target_lang.upper())
                kwargs = dict(source_lang=deepl_lang, target_lang=deepl_target)
                if context:
                    # Limit context to last 5 lines for API efficiency
                    recent_ctx = context[-5:] if len(context) > 5 else context
                    # Extract original text from (original, translation) tuples
                    recent_ctx = [c[0] if isinstance(c, tuple) else c for c in recent_ctx]
                    kwargs["context"] = "\n".join(recent_ctx)
                result = self.client.translate_text(text, **kwargs)
                translated = str(result) if result else None
                if translated:
                    self._cache_put(text, source_lang, translated)
                return translated
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)
                continue
        return None
