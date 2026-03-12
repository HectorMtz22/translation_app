"""Translation support using local Qwen model via MLX."""

import threading
from collections import OrderedDict

TRANSLATION_CACHE_SIZE = 256

QWEN_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

LANG_NAMES = {
    "ko": "Korean",
    "en": "English",
    "es": "Spanish",
}


class QwenTranslator:
    """Translates text using a local Qwen LLM with caching."""

    def __init__(self, target_lang="en", model_repo=QWEN_MODEL):
        self.target_lang = target_lang
        self._cache = OrderedDict()
        self._lock = threading.Lock()

        try:
            from mlx_lm import load

            print(f"Loading Qwen translator model '{model_repo}'...")
            self._model, self._tokenizer = load(model_repo)
            self._available = True
            print("Qwen translator model ready.")
        except Exception as e:
            print(f"[WARN] Failed to load Qwen model: {e}")
            print("       Install with: pip install mlx-lm")
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

        If context is provided (list of recent lines), it is included in the
        prompt to improve translation quality.
        Uses LRU cache to avoid re-translating identical text.
        """
        if not self._available:
            return None
        if source_lang == self.target_lang:
            return None

        if not context:
            cached = self._cache_get(text, source_lang)
            if cached is not None:
                return cached

        try:
            from mlx_lm import generate

            src_name = LANG_NAMES.get(source_lang, source_lang)
            tgt_name = LANG_NAMES.get(self.target_lang, self.target_lang)

            prompt_parts = [
                f"Translate the following {src_name} text to {tgt_name}.",
                "Output ONLY the translation, nothing else.",
            ]

            if context:
                recent_ctx = context[-5:] if len(context) > 5 else context
                prompt_parts.append(
                    f"\nContext (previous lines for reference, do NOT translate these):\n"
                    + "\n".join(recent_ctx)
                )

            prompt_parts.append(f"\nText to translate:\n{text}")

            messages = [{"role": "user", "content": "\n".join(prompt_parts)}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            with self._lock:
                result = generate(
                    self._model, self._tokenizer,
                    prompt=formatted,
                    max_tokens=512,
                    verbose=False,
                )

            translated = result.strip() if result else None
            if translated:
                self._cache_put(text, source_lang, translated)
            return translated
        except Exception:
            return None
