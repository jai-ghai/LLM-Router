"""On-screen text from a frame: logo bugs, tickers, headlines.

EasyOCR (Apache 2.0) with Devanagari + Latin by default; set GRADIST_OCR_LANGS (e.g. "ta,en")
for other scripts — EasyOCR only combines scripts that share a recogniser, so one reader per
script family. Lines under `min_confidence` are dropped: on broadcast graphics they are mostly
fragments of stylised text and add noise, not signal.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_LANGS = ("hi", "en")
MIN_CONFIDENCE = 0.3


@dataclass(frozen=True)
class ScreenLine:
    text: str
    confidence: float


def join_lines(lines: list[ScreenLine]) -> str:
    return " | ".join(line.text for line in lines)


def merge_frame_lines(per_frame: list[list[ScreenLine]]) -> list[ScreenLine]:
    """One list for several frames of the same clip: a repeated logo or ticker line is kept once
    (at its best confidence), in the order it first appeared."""
    best: dict[str, ScreenLine] = {}
    for lines in per_frame:
        for line in lines:
            key = " ".join(line.text.casefold().split())
            kept = best.get(key)
            if kept is None:
                best[key] = line
            elif line.confidence > kept.confidence:
                best[key] = ScreenLine(kept.text, line.confidence)
    return list(best.values())


class ScreenReader:
    """Lazy EasyOCR reader with a per-image cache (in memory, optionally mirrored to a JSON file
    so retraining does not re-read frames it has already seen)."""

    def __init__(
        self,
        langs: tuple[str, ...] = DEFAULT_LANGS,
        min_confidence: float = MIN_CONFIDENCE,
        cache_path: Path | None = None,
    ) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("easyocr is not installed. Install with: pip install .[ocr]") from exc
        self._easyocr = easyocr
        self.langs = langs
        self.min_confidence = min_confidence
        self.engine = f"EasyOCR ({'+'.join(langs)})"
        self._reader: Any = None
        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._cache: dict[str, list[dict[str, Any]]] = {}
        if cache_path is not None and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def read(self, image_bytes: bytes) -> list[ScreenLine]:
        key = hashlib.sha1(image_bytes).hexdigest()
        if key not in self._cache:
            with self._lock:
                # Re-check: a prefetch and a resolve can ask for the same frame at once; the
                # second waits here and then takes the cached result instead of reading again.
                if key not in self._cache:
                    self._cache[key] = [asdict(line) for line in self._run(image_bytes)]
        return [ScreenLine(**row) for row in self._cache[key]]

    def count_text(self, image_bytes: bytes) -> int:
        """How many text boxes the detector finds. Detection works for any script, so this
        measures how text-heavy a frame is even where the recogniser cannot read it."""
        import numpy as np
        from PIL import Image

        pixels = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        with self._lock:
            if self._reader is None:
                self._reader = self._easyocr.Reader(list(self.langs), gpu=False, verbose=False)
            horizontal, free = self._reader.detect(pixels)
        return len(horizontal[0]) + len(free[0])

    def save_cache(self) -> None:
        if self._cache_path is not None:
            self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    def _run(self, image_bytes: bytes) -> list[ScreenLine]:
        """Caller holds `self._lock` (EasyOCR's reader is not safe to share across threads)."""
        import numpy as np
        from PIL import Image

        pixels = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        if self._reader is None:
            self._reader = self._easyocr.Reader(list(self.langs), gpu=False, verbose=False)
        results = self._reader.readtext(pixels, detail=1, paragraph=False)
        return [
            ScreenLine(str(text).strip(), round(float(conf), 3))
            for _box, text, conf in results
            if float(conf) >= self.min_confidence and str(text).strip()
        ]
