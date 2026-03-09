"""Live rolling summarizer using a local MLX LLM."""

import multiprocessing as mp
import threading
import time

# Model to use for summarization (small, fast, multilingual)
SUMMARIZER_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# How many new transcript lines before triggering a new summary
SUMMARY_INTERVAL = 5


def _build_prompt(lines, target_lang):
    """Build the summarization prompt from transcript lines."""
    transcript = "\n".join(
        f"{l['speaker']}: {l['text']}" for l in lines
    )
    return (
        f"You are a summarizer. Below is a live conversation transcript "
        f"(may contain Korean, English, or Spanish). "
        f"Write a concise rolling summary in {target_lang}. "
        f"Focus on key topics, decisions, and important points. "
        f"Keep it under 200 words.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Summary:"
    )


def _generate_with_model(model, tokenizer, lines, target_lang):
    """Generate a summary using the loaded model."""
    from mlx_lm import generate

    prompt = _build_prompt(lines, target_lang)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    summary = generate(
        model, tokenizer,
        prompt=formatted,
        max_tokens=300,
        verbose=False,
    )
    return summary.strip()


class Summarizer:
    """Accumulates transcript lines and periodically generates a rolling summary.

    Runs summarization in-process (same MLX context as caller).
    """

    def __init__(self, target_lang="en", model_repo=SUMMARIZER_MODEL,
                 interval=SUMMARY_INTERVAL, on_summary=None):
        from mlx_lm import load

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

        self._last_summary = _generate_with_model(
            self._model, self._tokenizer, lines_snapshot, self.target_lang
        )
        return self._last_summary


class SummarizerProcess:
    """Runs summarization in a separate process for GPU independence.

    Uses multiprocessing with 'spawn' context to safely create a second
    MLX/Metal context. The child process loads its own copy of the model
    and generates summaries independently, so Whisper transcription in the
    main process is never blocked.

    Has the same public API as Summarizer (add_line, start, stop).
    """

    def __init__(self, target_lang="en", model_repo=SUMMARIZER_MODEL,
                 interval=SUMMARY_INTERVAL, on_summary=None):
        self.target_lang = target_lang
        self.model_repo = model_repo
        self.interval = interval
        self.on_summary = on_summary

        ctx = mp.get_context("spawn")
        self._line_queue = ctx.Queue()
        self._summary_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        self._process = ctx.Process(
            target=SummarizerProcess._worker,
            args=(self._line_queue, self._summary_queue, self._stop_event,
                  model_repo, target_lang, interval),
            daemon=True,
        )
        self._poll_thread = None

    def add_line(self, speaker, text, language):
        """Add a transcript line. Thread/process-safe via queue."""
        self._line_queue.put({"speaker": speaker, "text": text, "language": language})

    def start(self):
        """Start the child process and the polling thread."""
        self._process.start()
        self._poll_thread = threading.Thread(target=self._poll_summaries, daemon=True)
        self._poll_thread.start()

    def stop(self):
        """Signal the child to stop, wait for final summary, and clean up."""
        # Send sentinel to trigger final summary
        self._line_queue.put(None)
        self._process.join(timeout=45)
        if self._process.is_alive():
            self._process.terminate()

        # Drain any remaining summaries
        final = None
        while not self._summary_queue.empty():
            try:
                final = self._summary_queue.get_nowait()
            except Exception:
                break
        return final

    def _poll_summaries(self):
        """Poll the summary queue and invoke on_summary callback."""
        while not self._stop_event.is_set() or not self._summary_queue.empty():
            try:
                summary = self._summary_queue.get(timeout=1.0)
                if summary and self.on_summary:
                    self.on_summary(summary)
            except Exception:
                continue

    @staticmethod
    def _worker(line_queue, summary_queue, stop_event, model_repo, target_lang, interval):
        """Child process: loads model, generates summaries on demand."""
        from mlx_lm import load

        print(f"[SummarizerProcess] Loading model '{model_repo}'...")
        model, tokenizer = load(model_repo)
        print("[SummarizerProcess] Model ready.")

        lines = []
        lines_at_last_summary = 0

        while True:
            # Drain all available lines from the queue
            got_sentinel = False
            while True:
                try:
                    item = line_queue.get_nowait()
                    if item is None:
                        got_sentinel = True
                        break
                    lines.append(item)
                except Exception:
                    break

            pending = len(lines) - lines_at_last_summary

            if got_sentinel:
                # Final summary before exiting
                if lines:
                    summary = _generate_with_model(model, tokenizer, lines, target_lang)
                    summary_queue.put(summary)
                stop_event.set()
                break

            if pending >= interval:
                summary = _generate_with_model(model, tokenizer, list(lines), target_lang)
                lines_at_last_summary = len(lines)
                summary_queue.put(summary)

            time.sleep(1)
