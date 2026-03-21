"""Translation support using Meta's NLLB-200 model (local, offline)."""

import threading
from collections import OrderedDict

TRANSLATION_CACHE_SIZE = 256

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB uses BCP-47 style language codes
NLLB_LANG_CODES = {
    "ko": "kor_Hang",
    "en": "eng_Latn",
    "es": "spa_Latn",
}


class NLLBTranslator:
    """Translates text using Meta's NLLB-200 model locally with caching."""

    def __init__(self, target_lang="en", model_name=NLLB_MODEL):
        self.target_lang = target_lang
        self._cache = OrderedDict()
        self._lock = threading.Lock()

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tgt_code = NLLB_LANG_CODES.get(target_lang)
            if not self._tgt_code:
                raise ValueError(f"Unsupported target language: {target_lang}")

            print(f"Loading NLLB-200 model '{model_name}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._available = True
            print("NLLB-200 model ready.")
        except Exception as e:
            print(f"[WARN] Failed to load NLLB-200 model: {e}")
            print("       Install with: pip install transformers sentencepiece")
            self._model = None
            self._tokenizer = None
            self._available = False

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

        Context parameter is accepted for interface compatibility but NLLB
        is a seq2seq model that doesn't use conversational context.
        """
        if not self._available:
            return None
        if source_lang == self.target_lang:
            return None

        cached = self._cache_get(text, source_lang)
        if cached is not None:
            return cached

        src_code = NLLB_LANG_CODES.get(source_lang)
        if not src_code:
            return None

        try:
            self._tokenizer.src_lang = src_code
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            tgt_token_id = self._tokenizer.convert_tokens_to_ids(self._tgt_code)

            with self._lock:
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_token_id,
                    max_new_tokens=512,
                )

            translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if translated:
                self._cache_put(text, source_lang, translated)
                return translated
            return None
        except Exception:
            return None
