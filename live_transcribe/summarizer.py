"""Live rolling summarizer using a local MLX LLM."""

import threading
import time

from mlx_lm import load, generate

# Model to use for summarization (small, fast, multilingual)
SUMMARIZER_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# How many new transcript lines before triggering a new summary
SUMMARY_INTERVAL = 5


class Summarizer:
    """Accumulates transcript lines and periodically generates a rolling summary."""

    def __init__(self, target_lang="en", model_repo=SUMMARIZER_MODEL,
                 interval=SUMMARY_INTERVAL, on_summary=None):
        self.target_lang = target_lang
        self.interval = interval
        self.on_summary = on_summary  # callback(summary_text)

        self._lines = []
        self._lines_at_last_summary = 0
        self._last_summary = ""
        self._lock = threading.Lock()
        self._running = True
        self._thread = None

        print(f"Loading summarizer model '{model_repo}'...")
        self._model, self._tokenizer = load(model_repo)
        print("Summarizer model ready.")

    def add_line(self, speaker, text, language):
        """Add a transcript line. Thread-safe."""
        with self._lock:
            self._lines.append({"speaker": speaker, "text": text, "language": language})

    def start(self):
        """Start the background summarization thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background thread and return the final summary."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        # Generate one final summary
        return self._generate_summary()

    def _run(self):
        while self._running:
            time.sleep(1)
            with self._lock:
                pending = len(self._lines) - self._lines_at_last_summary
            if pending >= self.interval:
                summary = self._generate_summary()
                if summary and self.on_summary:
                    self.on_summary(summary)

    def _generate_summary(self):
        with self._lock:
            if not self._lines:
                return self._last_summary
            lines_snapshot = list(self._lines)
            self._lines_at_last_summary = len(self._lines)

        transcript = "\n".join(
            f"{l['speaker']}: {l['text']}" for l in lines_snapshot
        )

        prompt = (
            f"You are a summarizer. Below is a live conversation transcript "
            f"(may contain Korean, English, or Spanish). "
            f"Write a concise rolling summary in {self.target_lang}. "
            f"Focus on key topics, decisions, and important points. "
            f"Keep it under 200 words.\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Summary:"
        )

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        summary = generate(
            self._model, self._tokenizer,
            prompt=formatted,
            max_tokens=300,
            verbose=False,
        )

        self._last_summary = summary.strip()
        return self._last_summary
