"""Column-based display mode: side-by-side transcription and translation."""

from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

console = Console()


class ColumnsDisplay:
    """Two-column layout with Rich Live for in-place translation updates."""

    def __init__(self):
        self._last_speaker = None
        self._entries = []      # Renderables in the live area
        self._entry_map = {}    # entry_key -> index in _entries
        self._live = None
        # Keep entries bounded so Live doesn't exceed terminal height
        self._max_entries = max(5, (console.height - 5) // 3)

    def start(self):
        """Start the live display region."""
        self._live = Live(
            Group(*self._entries) if self._entries else Text(""),
            console=console,
            auto_refresh=False,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self):
        """Stop the live display, preserving content on screen."""
        if self._live:
            self._live.stop()
            self._live = None

    def _append(self, renderable, entry_key=None):
        """Add a renderable to the live display."""
        if self._live is None:
            console.print(renderable)
            return

        self._entries.append(renderable)
        if entry_key is not None:
            self._entry_map[entry_key] = len(self._entries) - 1

        # Commit oldest entries above the live area when buffer is full
        while len(self._entries) > self._max_entries:
            old = self._entries.pop(0)
            self._live.console.print(old)
            self._entry_map = {k: v - 1 for k, v in self._entry_map.items() if v > 0}

        self._refresh()

    def _refresh(self):
        if self._live and self._entries:
            self._live.update(Group(*self._entries))
            self._live.refresh()

    def print_segment_header(self, speaker, timestamp, has_translator,
                             entry_key=None):
        """Print speaker header when speaker changes."""
        if speaker == self._last_speaker:
            return
        self._last_speaker = speaker

        tw, indent, left_w, _ = self._get_col_widths()

        parts = [Text(f"\n{'─' * tw}", style="dim")]
        parts.append(Text(f"[{timestamp}] {speaker}:", style="bold cyan"))
        if has_translator:
            header = Text()
            header.append(" " * indent)
            header.append(self._pad_display("TRANSCRIPTION", left_w), style="dim bold")
            header.append(" │ ", style="dim")
            header.append("TRANSLATION", style="dim bold")
            parts.append(header)

        self._append(Group(*parts))

    def print_translated(self, speaker, text, translation, lang_tag,
                         timestamp=None, entry_key=None):
        """Print original (left) and translation (right)."""
        final_left = f"{text} [{lang_tag}]"
        right = f"→ {translation}" if translation else ""
        self._append(self._render_columns(final_left, right), entry_key=entry_key)

    def print_without_translation(self, speaker, text, lang_tag,
                                  timestamp=None, entry_key=None):
        """Print text (no translation)."""
        content = Text()
        content.append("  ")
        content.append(text, style="white")
        content.append(f"  [{lang_tag}]", style="dim")
        self._append(content, entry_key=entry_key)

    def update_translation(self, entry_key, speaker, text, new_translation,
                           lang_tag, timestamp=None):
        """Replace a previously printed translation in-place."""
        idx = self._entry_map.get(entry_key)
        if idx is None:
            return  # Entry already committed above live area

        final_left = f"{text} [{lang_tag}]"
        right = f"→ {new_translation}" if new_translation else ""
        self._entries[idx] = self._render_columns(
            final_left, right, right_style="bold green")
        self._refresh()

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
