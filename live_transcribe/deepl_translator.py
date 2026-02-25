"""English translation support using DeepL API."""

try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

import os
from dotenv import load_dotenv

load_dotenv()


class DeepLTranslator:
    """Translates text to English using DeepL API."""

    # Map short language codes to DeepL source language codes
    LANG_MAP = {
        "ko": "KO",
        "es": "ES",
        "en": "EN",
    }

    def __init__(self):
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

    def translate_to_english(self, text, source_lang):
        """Translate text to English. Returns None if already English or on failure."""
        if self.client is None:
            return None
        if source_lang == "en":
            return None
        try:
            deepl_lang = self.LANG_MAP.get(source_lang, source_lang.upper())
            result = self.client.translate_text(
                text, source_lang=deepl_lang, target_lang="EN-US"
            )
            return str(result) if result else None
        except Exception:
            return None
