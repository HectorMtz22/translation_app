#!/usr/bin/env python3
"""
Live System Audio Transcription with Speaker Diarization

Captures system audio via BlackHole virtual audio device,
transcribes using mlx-whisper (GPU-accelerated on Apple Silicon),
and separates speakers using resemblyzer voice embeddings.

Setup:
  1. Install BlackHole 2ch: brew install --cask blackhole-2ch
  2. Reboot Mac
  3. Create Multi-Output Device in Audio MIDI Setup
     (combine your speakers + BlackHole 2ch)
  4. Set Multi-Output Device as system output
  5. Run: ./live_transcribe_env/bin/python live_transcribe.py
"""

import sys
import signal
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
import mlx_whisper

# Try to import resemblyzer for speaker diarization
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("[WARN] resemblyzer not available - speaker separation disabled")

# ─── Configuration ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000          # Whisper expects 16kHz
CHUNK_DURATION = 6.0         # Seconds per transcription chunk (longer = more context)
OVERLAP_DURATION = 1.0       # Overlap between chunks for continuity
SILENCE_THRESHOLD = 0.005    # RMS threshold for silence detection
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"  # GPU-accelerated MLX model
SPEAKER_SIMILARITY = 0.75    # Cosine similarity threshold for same speaker
MAX_SPEAKERS = 10            # Maximum number of speakers to track
SUPPORTED_LANGUAGES = ["ko", "en", "es"]  # Korean, English, Spanish only

# ─── Global State ────────────────────────────────────────────────────────────

audio_buffer = deque(maxlen=int(SAMPLE_RATE * 60))  # 60s rolling buffer
chunk_buffer = []
running = True
lock = threading.Lock()


def find_blackhole_device():
    """Find the BlackHole 2ch input device index."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i, dev["name"]
    return None, None


def list_input_devices():
    """List all available input devices."""
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- BlackHole" if "blackhole" in dev["name"].lower() else ""
            print(f"  [{i}] {dev['name']} ({dev['max_input_channels']}ch, "
                  f"{dev['default_samplerate']:.0f}Hz){marker}")
    print()


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
                return best_match

            # New speaker detected
            if self.speaker_count < MAX_SPEAKERS:
                self.speaker_count += 1
                label = f"Speaker {self.speaker_count}"
                self.speaker_embeddings.append((label, embedding))
                return label
            else:
                # Too many speakers, assign to closest
                return best_match if best_match else "Speaker ?"

        except Exception as e:
            return self._last_speaker_label()

    def _last_speaker_label(self):
        if self.speaker_embeddings:
            return self.speaker_embeddings[-1][0]
        return "Speaker"


class LiveTranscriber:
    """Main live transcription engine."""

    def __init__(self, device_index, model_repo=WHISPER_MODEL):
        self.device_index = device_index
        self.speaker_tracker = SpeakerTracker()
        self.last_speaker = None
        self.transcript_lines = []
        self.model_repo = model_repo

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
        with lock:
            chunk_buffer.extend(audio.tolist())

    def process_audio(self):
        """Process accumulated audio chunks."""
        global running

        samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)
        overlap_samples = int(SAMPLE_RATE * OVERLAP_DURATION)

        while running:
            with lock:
                if len(chunk_buffer) < samples_per_chunk:
                    time.sleep(0.1)
                    continue
                # Extract chunk with overlap
                audio_data = np.array(chunk_buffer[:samples_per_chunk], dtype=np.float32)
                # Keep overlap for continuity
                del chunk_buffer[:samples_per_chunk - overlap_samples]

            # Skip silence
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < SILENCE_THRESHOLD:
                continue

            # Transcribe
            try:
                # First pass: detect language from supported set
                detect_result = mlx_whisper.transcribe(
                    audio_data, path_or_hf_repo=self.model_repo,
                    temperature=0.0,
                )
                detected = detect_result["language"] if detect_result["language"] in SUPPORTED_LANGUAGES else "en"

                # Second pass: transcribe with locked language for accuracy
                # Uses greedy decoding (temp=0) with fallback temperatures
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo=self.model_repo,
                    language=detected,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                )

                lang = result["language"] or "??"

                for segment in result["segments"]:
                    text = segment["text"].strip()
                    if not text or len(text) < 2:
                        continue

                    # Get the audio portion for this segment
                    start_sample = int(segment["start"] * SAMPLE_RATE)
                    end_sample = min(int(segment["end"] * SAMPLE_RATE), len(audio_data))
                    segment_audio = audio_data[start_sample:end_sample]

                    # Identify speaker
                    speaker = self.speaker_tracker.identify_speaker(segment_audio)

                    # Format and print
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    if speaker != self.last_speaker:
                        # New speaker - print header
                        print(f"\n\033[1;36m[{timestamp}] {speaker}:\033[0m")
                        self.last_speaker = speaker

                    # Print transcribed text
                    print(f"  \033[0;37m{text}\033[0m  \033[0;90m[{lang}]\033[0m")

                    self.transcript_lines.append({
                        "time": timestamp,
                        "speaker": speaker,
                        "text": text,
                        "language": lang,
                    })

            except Exception as e:
                print(f"\033[0;31m[Error] Transcription failed: {e}\033[0m")

    def save_transcript(self):
        """Save transcript to file."""
        if not self.transcript_lines:
            return

        import os
        transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(transcript_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            current_speaker = None
            for line in self.transcript_lines:
                if line["speaker"] != current_speaker:
                    current_speaker = line["speaker"]
                    f.write(f"\n[{line['time']}] {current_speaker}:\n")
                f.write(f"  {line['text']}\n")
        print(f"\n\033[1;32mTranscript saved to: {filepath}\033[0m")

    def start(self):
        """Start live transcription."""
        global running

        print("=" * 60)
        print("  LIVE TRANSCRIPTION WITH SPEAKER DIARIZATION")
        print("=" * 60)
        print(f"  Audio device: {sd.query_devices(self.device_index)['name']}")
        print(f"  Whisper model: {WHISPER_MODEL}")
        print(f"  Chunk duration: {CHUNK_DURATION}s")
        print(f"  Speaker diarization: {'ON' if DIARIZATION_AVAILABLE else 'OFF'}")
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
            blocksize=int(SAMPLE_RATE * 0.5),  # 500ms blocks
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
            self.save_transcript()
            print("Done.")


def main():
    print("\n\033[1mLive Transcribe - System Audio\033[0m\n")

    # Find BlackHole device
    device_idx, device_name = find_blackhole_device()

    if device_idx is None:
        print("\033[0;31mBlackHole device not found!\033[0m")
        print("\nMake sure you have:")
        print("  1. Installed BlackHole: brew install --cask blackhole-2ch")
        print("  2. Rebooted your Mac")
        print("  3. Created a Multi-Output Device in Audio MIDI Setup")
        print("     (combining your speakers + BlackHole 2ch)")
        print("  4. Set the Multi-Output Device as system output")
        list_input_devices()

        # Allow manual device selection
        try:
            choice = input("Enter device index to use (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            device_idx = int(choice)
        except (ValueError, EOFError):
            sys.exit(1)
    else:
        print(f"Found BlackHole device: [{device_idx}] {device_name}")

    # Start transcription
    transcriber = LiveTranscriber(device_idx)
    transcriber.start()


if __name__ == "__main__":
    main()
