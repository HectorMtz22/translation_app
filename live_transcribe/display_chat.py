"""Chat bubble display mode: messages appear as aligned chat bubbles per speaker."""

import time

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()

FLUID_WORD_DELAY = 0.04

SPEAKER_COLORS = ["cyan", "green", "magenta", "yellow", "blue"]


class ChatDisplay:
    """Chat-style layout with bubbles aligned left/right per speaker."""

    def __init__(self):
        self._side_map = {}
        self._next_side = 0
        self._color_map = {}
        self._color_idx = 0

    def print_segment_header(self, speaker, timestamp, has_translator):
        """No-op for chat mode — speaker info is shown in the bubble title."""
        pass

    def print_with_translation(self, speaker, text, translation_future, lang_tag,
                               timestamp=None, delay=FLUID_WORD_DELAY):
        """Print a chat bubble with fluid animation for translation only."""
        with Live(self._render_bubble(speaker, text, translation="…",
                                      lang_tag=lang_tag, timestamp=timestamp),
                  console=console, refresh_per_second=25,
                  transient=False) as live:
            translation = None
            try:
                translation = translation_future.result(timeout=10.0)
            except Exception:
                pass

            if translation:
                right_words = translation.split()
                built_r = ""
                for word in right_words:
                    built_r += (" " if built_r else "") + word
                    live.update(self._render_bubble(speaker, text, translation=f"→ {built_r}",
                                                   lang_tag=lang_tag, timestamp=timestamp))
                    time.sleep(delay)
                live.update(self._render_bubble(speaker, text, translation=f"→ {translation}",
                                               lang_tag=lang_tag, timestamp=timestamp))
            else:
                live.update(self._render_bubble(speaker, text, lang_tag=lang_tag,
                                               timestamp=timestamp))

        return translation

    def print_without_translation(self, speaker, text, lang_tag,
                                  timestamp=None, delay=FLUID_WORD_DELAY):
        """Print a chat bubble instantly (no translation)."""
        console.print(self._render_bubble(speaker, text, lang_tag=lang_tag,
                                          timestamp=timestamp))

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _speaker_side(self, speaker):
        if speaker not in self._side_map:
            self._side_map[speaker] = "left" if self._next_side % 2 == 0 else "right"
            self._next_side += 1
        return self._side_map[speaker]

    def _speaker_color(self, speaker):
        if speaker not in self._color_map:
            self._color_map[speaker] = SPEAKER_COLORS[self._color_idx % len(SPEAKER_COLORS)]
            self._color_idx += 1
        return self._color_map[speaker]

    def _render_bubble(self, speaker, text, translation=None, lang_tag=None, timestamp=None):
        side = self._speaker_side(speaker)
        color = self._speaker_color(speaker)
        bubble_width = min(console.width * 2 // 3, 60)

        body = Text()
        body.append(text, style="white")
        if lang_tag:
            body.append(f"  [{lang_tag}]", style="dim")
        if translation:
            body.append(f"\n{translation}", style="yellow")

        title = f"{speaker}" + (f"  {timestamp}" if timestamp else "")
        panel = Panel(
            body,
            title=title,
            title_align="left",
            border_style=color,
            width=bubble_width,
            padding=(0, 1),
        )
        return Align(panel, align=side)
