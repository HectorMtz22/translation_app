"""Translation support using DeepL API."""

try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

import os
from dotenv import load_dotenv

load_dotenv()


class DeepLTranslator:
    """Translates text using DeepL API."""

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

    def translate(self, text, source_lang):
        """Translate text to target language. Returns None if same language or on failure."""
        if self.client is None:
            return None
        if source_lang == self.target_lang:
            return None
        try:
            deepl_lang = self.LANG_MAP.get(source_lang, source_lang.upper())
            deepl_target = self.TARGET_LANG_MAP.get(self.target_lang, self.target_lang.upper())
            result = self.client.translate_text(
                text, source_lang=deepl_lang, target_lang=deepl_target
            )
            return str(result) if result else None
        except Exception:
            return None
