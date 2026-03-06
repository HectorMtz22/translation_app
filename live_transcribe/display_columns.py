"""Column-based display mode: side-by-side transcription and translation."""

import time

from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.text import Text

console = Console()

FLUID_WORD_DELAY = 0.04


class ColumnsDisplay:
    """Two-column layout with transcription on the left and translation on the right."""

    def __init__(self):
        self._last_speaker = None

    def print_segment_header(self, speaker, timestamp, has_translator):
        """Print speaker header when speaker changes."""
        if speaker == self._last_speaker:
            return
        self._last_speaker = speaker

        tw, indent, left_w, right_w = self._get_col_widths()
        console.print(f"\n[dim]{'─' * tw}[/dim]")
        console.print(f"[bold cyan][{timestamp}] {speaker}:[/bold cyan]")
        if has_translator:
            header = Text()
            header.append(" " * indent)
            header.append(self._pad_display("TRANSCRIPTION", left_w), style="dim bold")
            header.append(" │ ", style="dim")
            header.append("TRANSLATION", style="dim bold")
            console.print(header)

    def print_with_translation(self, speaker, text, translation_future, lang_tag,
                               timestamp=None, delay=FLUID_WORD_DELAY):
        """Print original (left) and translation (right) with fluid animation on translation only."""
        final_left = f"{text} [{lang_tag}]"

        with Live(self._render_columns(final_left, "…"),
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
                    live.update(self._render_columns(final_left, f"→ {built_r}"))
                    time.sleep(delay)
                live.update(self._render_columns(final_left, f"→ {translation}"))
            else:
                live.update(self._render_columns(final_left, ""))

        return translation

    def print_without_translation(self, speaker, text, lang_tag,
                                  timestamp=None, delay=FLUID_WORD_DELAY):
        """Print text instantly (no translation)."""
        console.print(f"  [white]{text}[/white]  [dim][{lang_tag}][/dim]")

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _get_col_widths(self):
        tw = console.width
        sep_len = 3  # " │ "
        indent = 2
        usable = tw - indent - sep_len
        left_w = usable // 2
        right_w = usable - left_w
        return tw, indent, left_w, right_w

    @staticmethod
    def _wrap_display(text, width):
        if not text:
            return [""]
        words = text.split()
        lines = []
        line = ""
        line_w = 0
        for word in words:
            w = cell_len(word)
            if line:
                if line_w + 1 + w <= width:
                    line += " " + word
                    line_w += 1 + w
                else:
                    lines.append(line)
                    line = word
                    line_w = w
            else:
                line = word
                line_w = w
        if line:
            lines.append(line)
        return lines or [""]

    @staticmethod
    def _pad_display(text, width):
        pad = width - cell_len(text)
        return text + " " * max(pad, 0)

    def _render_columns(self, left_str, right_str, left_style="white", right_style="yellow"):
        _, indent, left_w, right_w = self._get_col_widths()

        left_lines = self._wrap_display(left_str, left_w)
        right_lines = self._wrap_display(right_str, right_w)
        n = max(len(left_lines), len(right_lines))
        left_lines += [""] * (n - len(left_lines))
        right_lines += [""] * (n - len(right_lines))

        result = Text()
        for i, (l, r) in enumerate(zip(left_lines, right_lines)):
            result.append(" " * indent)
            result.append(self._pad_display(l, left_w), style=left_style)
            result.append(" │ ", style="dim")
            result.append(r, style=right_style)
            if i < n - 1:
                result.append("\n")
        return result
