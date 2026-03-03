#!/usr/bin/env python3
"""
Live System Audio Transcription with Speaker Diarization

Captures system audio via BlackHole virtual audio device,
transcribes using mlx-whisper (GPU-accelerated on Apple Silicon),
and separates speakers using resemblyzer voice embeddings.

Uses Silero VAD for voice activity detection to trigger transcription
on natural speech boundaries instead of fixed time windows.

Setup:
  1. Install BlackHole 2ch: brew install --cask blackhole-2ch
  2. Reboot Mac
  3. Create Multi-Output Device in Audio MIDI Setup
     (combine your speakers + BlackHole 2ch)
  4. Set Multi-Output Device as system output
  5. Run: ./live_transcribe_env/bin/python live_transcribe.py
"""

import argparse
import os
import sys
import signal
import textwrap
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import sounddevice as sd
import mlx_whisper
import torch

from translator import Translator
from deepl_translator import DeepLTranslator

# Try to import resemblyzer for speaker diarization
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("[WARN] resemblyzer not available - speaker separation disabled")

# ─── Configuration ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000          # Whisper expects 16kHz
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"  # GPU-accelerated MLX model
SPEAKER_SIMILARITY = 0.82    # Cosine similarity threshold for same speaker
NUM_SPEAKERS = 5             # Expected number of speakers (once reached, assigns to closest match)
MAX_SPEAKERS = 5             # Maximum number of speakers to track
MIN_CHUNKS_NEW_SPEAKER = 2   # Require N consecutive unmatched chunks before creating a new speaker
SUPPORTED_LANGUAGES = ["ko", "en", "es"]  # Korean, English, Spanish only
LANG_NAMES = {"ko": "Korean", "en": "English", "es": "Spanish"}

# ─── VAD Configuration ──────────────────────────────────────────────────────

VAD_THRESHOLD = 0.3          # Speech probability threshold (lower = more sensitive)
MIN_SPEECH_DURATION = 0.3    # Ignore speech segments shorter than this (seconds)
MAX_SPEECH_DURATION = 15.0   # Force transcription after this much continuous speech (seconds)
SILENCE_AFTER_SPEECH = 0.5   # Pause duration to trigger end-of-speech (seconds)
VAD_FRAME_SAMPLES = 512      # Silero VAD frame size (512 samples = 32ms at 16kHz)
ENERGY_THRESHOLD = 0.002     # Minimum RMS energy to consider as real speech (not background noise)
FLUID_WORD_DELAY = 0.04      # Seconds between words for fluid typewriter printing

# ─── Global State ────────────────────────────────────────────────────────────

running = True
lock = threading.Lock()


def find_blackhole_device():
    """Find the BlackHole 2ch input device index."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i, dev["name"]
    return None, None


def list_input_devices(default_idx=None):
    """List all available input devices, highlighting the default."""
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            markers = []
            if "blackhole" in dev["name"].lower():
                markers.append("BlackHole")
            if i == default_idx:
                markers.append("default")
            suffix = f"  <-- [{', '.join(markers)}]" if markers else ""
            print(f"  [{i}] {dev['name']} ({dev['max_input_channels']}ch, "
                  f"{dev['default_samplerate']:.0f}Hz){suffix}")
    print()


def load_vad_model():
    """Load Silero VAD model from PyTorch Hub."""
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    print("VAD model ready.")
    return model


class SpeakerTracker:
    """Track and identify speakers using voice embeddings."""

    def __init__(self):
        if DIARIZATION_AVAILABLE:
            print("Loading speaker encoder model...")
            self.encoder = VoiceEncoder()
            print("Speaker encoder ready.")
        else:
            self.encoder = None
        self.speaker_embeddings = []  # List of (label, embedding) tuples
        self.speaker_count = 0
        self.unmatched_streak = 0      # Consecutive chunks that didn't match any speaker
        self.pending_embedding = None  # Embedding accumulator for potential new speaker

    def identify_speaker(self, audio_chunk):
        """Identify or register a speaker from an audio chunk."""
        if not DIARIZATION_AVAILABLE or self.encoder is None:
            return "Speaker"

        if len(audio_chunk) < SAMPLE_RATE * 0.5:  # Need at least 0.5s
            return self._last_speaker_label()

        try:
            # Preprocess and get embedding
            processed = preprocess_wav(audio_chunk, source_sr=SAMPLE_RATE)
            if len(processed) < SAMPLE_RATE * 0.3:
                return self._last_speaker_label()

            embedding = self.encoder.embed_utterance(processed)

            # Compare against known speakers
            best_match = None
            best_similarity = -1

            for label, known_emb in self.speaker_embeddings:
                similarity = np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-8
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = label

            # If good match found, update the embedding (rolling average)
            if best_match and best_similarity >= SPEAKER_SIMILARITY:
                for i, (label, known_emb) in enumerate(self.speaker_embeddings):
                    if label == best_match:
                        # Exponential moving average of embeddings
                        self.speaker_embeddings[i] = (
                            label,
                            0.8 * known_emb + 0.2 * embedding,
                        )
                        break
                self.unmatched_streak = 0
                self.pending_embedding = None
                return best_match

            # No good match — decide whether to create a new speaker
            # If we've already reached the expected number of speakers, assign to closest
            if self.speaker_count >= NUM_SPEAKERS:
                return best_match if best_match else "Speaker ?"

            # Require multiple consecutive unmatched chunks before creating a new speaker
            self.unmatched_streak += 1
            if self.pending_embedding is None:
                self.pending_embedding = embedding
            else:
                self.pending_embedding = 0.5 * self.pending_embedding + 0.5 * embedding

            if self.unmatched_streak >= MIN_CHUNKS_NEW_SPEAKER:
                if self.speaker_count < MAX_SPEAKERS:
                    self.speaker_count += 1
                    label = f"Speaker {self.speaker_count}"
                    self.speaker_embeddings.append((label, self.pending_embedding))
                    self.unmatched_streak = 0
                    self.pending_embedding = None
                    return label

            # Not enough evidence yet for a new speaker — assign to closest existing
            if best_match:
                return best_match
            return self._last_speaker_label()

        except Exception as e:
            return self._last_speaker_label()

    def _last_speaker_label(self):
        if self.speaker_embeddings:
            return self.speaker_embeddings[-1][0]
        return "Speaker"


class LiveTranscriber:
    """Main live transcription engine with VAD-driven segmentation."""

    def __init__(self, device_index, translator=None, translate_langs=None,
                 target_lang="en", model_repo=WHISPER_MODEL):
        self.device_index = device_index
        self.speaker_tracker = SpeakerTracker()
        self.translator = translator
        self.translate_langs = translate_langs or set()
        self.target_lang = target_lang
        self.last_speaker = None
        self.transcript_lines = []
        self.model_repo = model_repo
        self.print_lock = threading.Lock()
        self.transcription_pool = ThreadPoolExecutor(max_workers=1)
        self.translation_pool = ThreadPoolExecutor(max_workers=4) if translator else None

        # VAD state
        self.vad_model = load_vad_model()
        self.audio_queue = deque()  # Raw audio frames waiting for VAD processing
        self.speech_buffer = []     # Accumulated audio during speech
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.silence_start_time = 0.0
        self.vad_lock = threading.Lock()

        print(f"Loading Whisper model '{model_repo}' (GPU-accelerated)...")
        # Warm up the model by running a dummy transcription
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(dummy, path_or_hf_repo=model_repo, language="en", temperature=0.0)
        print("Whisper model ready.\n")

    def audio_callback(self, indata, frames, time_info, status):
        """Called for each audio block from sounddevice."""
        if status:
            pass  # Ignore overflow warnings silently
        audio = indata[:, 0].copy()  # Mono
        with self.vad_lock:
            self.audio_queue.append(audio)

    def process_audio(self):
        """Process audio through VAD and trigger transcription on speech boundaries."""
        global running

        while running:
            # Grab any queued audio
            with self.vad_lock:
                if not self.audio_queue:
                    time.sleep(0.01)
                    continue
                chunks = list(self.audio_queue)
                self.audio_queue.clear()

            # Concatenate all queued audio into one array
            raw_audio = np.concatenate(chunks)

            # Process in VAD_FRAME_SAMPLES-sized frames
            offset = 0
            while offset + VAD_FRAME_SAMPLES <= len(raw_audio):
                frame = raw_audio[offset:offset + VAD_FRAME_SAMPLES]
                offset += VAD_FRAME_SAMPLES

                # Run VAD on this frame
                frame_tensor = torch.from_numpy(frame).float()
                speech_prob = self.vad_model(frame_tensor, SAMPLE_RATE).item()

                now = time.monotonic()

                if speech_prob >= VAD_THRESHOLD:
                    # Speech detected
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speech_start_time = now
                    self.silence_start_time = 0.0
                    self.speech_buffer.append(frame)

                    # Force transcription if speech is too long
                    speech_duration = now - self.speech_start_time
                    if speech_duration >= MAX_SPEECH_DURATION:
                        self._flush_speech_buffer()

                else:
                    # Silence / non-speech
                    if self.is_speaking:
                        # Still accumulate audio during short pauses
                        self.speech_buffer.append(frame)

                        if self.silence_start_time == 0.0:
                            self.silence_start_time = now
                        elif now - self.silence_start_time >= SILENCE_AFTER_SPEECH:
                            # Enough silence after speech — trigger transcription
                            self._flush_speech_buffer()

            # Keep any leftover samples for the next iteration
            remainder = len(raw_audio) - offset
            if remainder > 0:
                with self.vad_lock:
                    self.audio_queue.appendleft(raw_audio[offset:])

    def _flush_speech_buffer(self):
        """Send accumulated speech buffer to transcription and reset VAD state."""
        if not self.speech_buffer:
            self.is_speaking = False
            self.silence_start_time = 0.0
            return

        audio_data = np.concatenate(self.speech_buffer).astype(np.float32)

        # Reset VAD state
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_start_time = 0.0
        self.vad_model.reset_states()

        # Check minimum speech duration
        duration = len(audio_data) / SAMPLE_RATE
        if duration < MIN_SPEECH_DURATION:
            return

        # Check energy level — reject quiet noise that slipped past VAD
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < ENERGY_THRESHOLD:
            return

        self.transcription_pool.submit(self._transcribe_segment, audio_data)

    @staticmethod
    def _is_hallucination(text):
        """Detect Whisper hallucination patterns (repetitive tokens or phrases)."""
        stripped = text.strip()
        if not stripped:
            return True
        # Split into words/tokens
        tokens = stripped.split()
        if len(tokens) < 3:
            # For very short text, check if it's just repeated characters
            # e.g. "와와" or "aaaa"
            unique_chars = set(stripped.replace(" ", ""))
            if len(unique_chars) <= 2 and len(stripped) > 1:
                return True
            return False
        # Check if most tokens are the same (repetitive hallucination)
        unique_tokens = set(tokens)
        if len(unique_tokens) <= 2 and len(tokens) >= 4:
            return True
        # Check if any single token dominates (>70% of all tokens)
        counts = Counter(tokens)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(tokens) > 0.7 and len(tokens) >= 4:
            return True
        # Detect repeating n-gram phrases (e.g. "A B C A B C A B C")
        for n in range(2, min(len(tokens) // 2 + 1, 8)):
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            ngram_counts = Counter(ngrams)
            most_common_ngram, mc_count = ngram_counts.most_common(1)[0]
            if mc_count >= 3 and (mc_count * len(most_common_ngram)) / len(tokens) > 0.5:
                return True
        return False

    def _print_fluid_line(self, words, format_fn, delay=FLUID_WORD_DELAY):
        """Print words progressively, rewriting the line for a typewriter effect.

        Must be called while holding self.print_lock.
        """
        built = ""
        for i, word in enumerate(words):
            if built:
                built += " "
            built += word
            sys.stdout.write(f"\r{format_fn(built)}")
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _transcribe_segment(self, audio_data):
        """Transcribe an audio segment with single-pass Whisper."""
        try:
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=self.model_repo,
                temperature=(0.0, 0.4),
                condition_on_previous_text=False,
                compression_ratio_threshold=1.8,
                logprob_threshold=-1.0,
                no_speech_threshold=0.8,
            )

            lang = result.get("language", "??")
            if lang not in SUPPORTED_LANGUAGES:
                # Whisper detected an unsupported language — likely misdetection,
                # discard the entire result since the transcription itself is wrong
                return

            for segment in result["segments"]:
                text = segment["text"].strip()
                if not text or len(text) < 2:
                    continue

                # Filter low-confidence segments using per-segment metrics
                avg_logprob = segment.get("avg_logprob", 0)
                no_speech_prob = segment.get("no_speech_prob", 0)
                if avg_logprob < -1.0 or no_speech_prob > 0.7:
                    continue

                # Filter out Whisper hallucinations
                if self._is_hallucination(text):
                    continue

                # Get the audio portion for this segment for speaker ID
                start_sample = int(segment["start"] * SAMPLE_RATE)
                end_sample = min(int(segment["end"] * SAMPLE_RATE), len(audio_data))
                segment_audio = audio_data[start_sample:end_sample]

                # Identify speaker
                speaker = self.speaker_tracker.identify_speaker(segment_audio)

                # Format and print
                timestamp = datetime.now().strftime("%H:%M:%S")

                if speaker != self.last_speaker:
                    # New speaker - print header with separator line
                    try:
                        tw = os.get_terminal_size().columns
                    except OSError:
                        tw = 120
                    with self.print_lock:
                        print(f"\n\033[0;90m{'─' * tw}\033[0m")
                        print(f"\033[1;36m[{timestamp}] {speaker}:\033[0m")
                    self.last_speaker = speaker

                # Store transcript entry (translation filled async)
                entry = {
                    "time": timestamp,
                    "speaker": speaker,
                    "text": text,
                    "language": lang,
                    "translation": None,
                }
                self.transcript_lines.append(entry)

                # Print original text with fluid word-by-word effect
                if self.translator and lang in self.translate_langs:
                    # Fire translation NOW so it runs during original text animation
                    future = self.translation_pool.submit(
                        self.translator.translate, entry["text"], entry["language"]
                    )

                    with self.print_lock:
                        # Print original text
                        self._print_fluid_line(
                            f"{text} [{lang}]".split(),
                            lambda b: f"  \033[0;37m{b}\033[0m",
                        )
                        # Translation was running in parallel — grab result
                        try:
                            translation = future.result(timeout=5.0)
                        except Exception:
                            translation = None
                        if translation:
                            entry["translation"] = translation
                            self._print_fluid_line(
                                translation.split(),
                                lambda b: f"  \033[0;33m→ {b}\033[0m",
                            )
                        print()
                else:
                    with self.print_lock:
                        self._print_fluid_line(
                            text.split(),
                            lambda b, lg=lang: f"  \033[0;37m{b}\033[0m  \033[0;90m[{lg}]\033[0m",
                        )

        except Exception as e:
            print(f"\033[0;31m[Error] Transcription failed: {e}\033[0m")

    def save_transcript(self):
        """Save transcript to separate original and English translation files."""
        if not self.transcript_lines:
            return

        transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        header_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save original transcript
        original_path = os.path.join(transcript_dir, f"transcript_{timestamp}_original.txt")
        with open(original_path, "w") as f:
            f.write(f"Transcript (Original) - {header_time}\n")
            f.write("=" * 60 + "\n\n")
            current_speaker = None
            for line in self.transcript_lines:
                if line["speaker"] != current_speaker:
                    current_speaker = line["speaker"]
                    f.write(f"\n[{line['time']}] {current_speaker}:\n")
                f.write(f"  {line['text']}\n")

        # Save translated transcript
        has_translations = any(line.get("translation") for line in self.transcript_lines)
        if has_translations:
            target_name = LANG_NAMES.get(self.target_lang, self.target_lang).lower()
            translated_path = os.path.join(transcript_dir, f"transcript_{timestamp}_{target_name}.txt")
            target_label = LANG_NAMES.get(self.target_lang, self.target_lang)
            with open(translated_path, "w") as f:
                f.write(f"Transcript ({target_label}) - {header_time}\n")
                f.write("=" * 60 + "\n\n")
                current_speaker = None
                for line in self.transcript_lines:
                    if line["speaker"] != current_speaker:
                        current_speaker = line["speaker"]
                        f.write(f"\n[{line['time']}] {current_speaker}:\n")
                    f.write(f"  {line.get('translation') or line['text']}\n")
            print(f"\n\033[1;32mTranscripts saved to:\n  {original_path}\n  {translated_path}\033[0m")
        else:
            print(f"\n\033[1;32mTranscript saved to: {original_path}\033[0m")

    def start(self):
        """Start live transcription."""
        global running

        print("=" * 60)
        print("  LIVE TRANSCRIPTION WITH SPEAKER DIARIZATION")
        print("=" * 60)
        print(f"  Audio device: {sd.query_devices(self.device_index)['name']}")
        print(f"  Whisper model: {WHISPER_MODEL}")
        print(f"  VAD threshold: {VAD_THRESHOLD}")
        print(f"  Silence trigger: {SILENCE_AFTER_SPEECH}s")
        print(f"  Max speech segment: {MAX_SPEECH_DURATION}s")
        print(f"  Speaker diarization: {'ON' if DIARIZATION_AVAILABLE else 'OFF'}")
        if self.translator:
            from_list = ", ".join(
                f"{LANG_NAMES.get(l, l)} ({l})" for l in sorted(self.translate_langs)
            )
            to_name = f"{LANG_NAMES.get(self.target_lang, self.target_lang)} ({self.target_lang})"
            print(f"  Translation: ON")
            print(f"  Translate from: {from_list}")
            print(f"  Translate to:   {to_name}")
        else:
            print(f"  Translation: OFF")
        print("=" * 60)
        print("  Press Ctrl+C to stop and save transcript")
        print("=" * 60)
        print("\nListening...\n")

        # Start audio stream
        stream = sd.InputStream(
            device=self.device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks (smaller for lower latency)
            callback=self.audio_callback,
        )

        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)

        def signal_handler(sig, frame):
            global running
            running = False
            print("\n\nStopping...")

        signal.signal(signal.SIGINT, signal_handler)

        try:
            stream.start()
            process_thread.start()

            while running:
                time.sleep(0.1)

        finally:
            stream.stop()
            stream.close()
            running = False
            process_thread.join(timeout=5)
            self.transcription_pool.shutdown(wait=True, cancel_futures=False)
            if self.translation_pool:
                self.translation_pool.shutdown(wait=True, cancel_futures=False)
            self.save_transcript()
            print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Live system audio transcription with speaker diarization")
    parser.add_argument("-d", "--device", type=int, default=None,
                        help="Input device index (skip interactive prompt)")
    parser.add_argument("-t", "--translator", choices=["google", "deepl", "none"], default=None,
                        help="Translation service: google, deepl, or none to disable")
    parser.add_argument("--translate-from", default=None,
                        help="Comma-separated source language codes to translate (default: ko)")
    parser.add_argument("--translate-to", default=None,
                        help="Target language code (default: en)")
    args = parser.parse_args()

    print("\n\033[1mLive Transcribe - System Audio\033[0m\n")

    if args.device is not None:
        # CLI flag provided — use directly
        device_idx = args.device
        dev_info = sd.query_devices(device_idx)
        print(f"Using device: [{device_idx}] {dev_info['name']}")
    else:
        # Determine default: prefer BlackHole, fall back to system default input
        bh_idx, bh_name = find_blackhole_device()
        if bh_idx is not None:
            default_idx = bh_idx
        else:
            default_idx = sd.default.device[0]  # system default input

        list_input_devices(default_idx)

        try:
            choice = input(f"Select device index [Enter={default_idx}]: ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            device_idx = int(choice) if choice else default_idx
        except (ValueError, EOFError):
            sys.exit(1)

    # Select translation service
    if args.translator is not None:
        translator_choice = args.translator
    else:
        print("\nTranslation service:")
        print("-" * 40)
        print("  [1] Google Translate")
        print("  [2] DeepL")
        print("  [3] None (transcription only)")
        print()
        try:
            t_choice = input("Select translation service [Enter=1]: ").strip()
            if t_choice == "2":
                translator_choice = "deepl"
            elif t_choice == "3":
                translator_choice = "none"
            else:
                translator_choice = "google"
        except (ValueError, EOFError):
            translator_choice = "google"

    # Defaults for source/target languages
    translate_langs = set()
    target_lang = "en"

    if translator_choice == "none":
        translator = None
        print("Using translator: None (disabled)")
    else:
        # Determine source languages to translate
        if args.translate_from is not None:
            if args.translate_from == "all":
                translate_langs = set(LANG_NAMES.keys())
            else:
                translate_langs = set(args.translate_from.split(","))
        else:
            # Interactive multi-select
            lang_options = list(LANG_NAMES.items())
            print("\nTranslate FROM (comma-separated, Enter=1):")
            for i, (code, name) in enumerate(lang_options, 1):
                print(f"  [{i}] {name} ({code})")
            print(f"  [*] All")
            try:
                choice = input("Select: ").strip()
                if choice == "*":
                    translate_langs = {code for code, _ in lang_options}
                elif choice:
                    for idx_str in choice.split(","):
                        idx = int(idx_str.strip()) - 1
                        if 0 <= idx < len(lang_options):
                            translate_langs.add(lang_options[idx][0])
                else:
                    translate_langs = {"ko"}  # default
            except (ValueError, EOFError):
                translate_langs = {"ko"}

        # Determine target language
        if args.translate_to is not None:
            target_lang = args.translate_to
        else:
            # Interactive single-select
            lang_options = list(LANG_NAMES.items())
            print("\nTranslate TO:")
            for i, (code, name) in enumerate(lang_options, 1):
                print(f"  [{i}] {name} ({code})")
            try:
                choice = input("Select [Enter=1]: ").strip()
                if choice:
                    idx = int(choice) - 1
                    if 0 <= idx < len(lang_options):
                        target_lang = lang_options[idx][0]
                else:
                    target_lang = "en"  # default
            except (ValueError, EOFError):
                target_lang = "en"

        # Remove target language from source set (no point translating to itself)
        translate_langs.discard(target_lang)

        if translator_choice == "deepl":
            translator = DeepLTranslator(target_lang=target_lang)
            print(f"Using translator: DeepL")
        else:
            translator = Translator(target_lang=target_lang)
            print(f"Using translator: Google Translate")

    # Start transcription
    transcriber = LiveTranscriber(
        device_idx, translator=translator,
        translate_langs=translate_langs, target_lang=target_lang,
    )
    transcriber.start()


if __name__ == "__main__":
    main()
