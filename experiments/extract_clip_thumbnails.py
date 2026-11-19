# [Fact-Forcing preamble]
#
# Callers: CLI — `python experiments/extract_clip_thumbnails.py`; also imported by
# experiments/train_from_video_clips.py for extract-then-train.
# Existing files: UI ImagePicker uploads stills only; BARC README notes production has no
# thumbnail extractor; no other script pulls frames from clips into data/images/.
# Reads: video files under data/video_clips/ (mp4/mov/mkv/webm/avi). Writes: PNG thumbnails
# under data/thumbnails/ (one per clip) and prints a JSON summary to stdout.
# User: "ha yeh kar lo ..mai video clips dunga or tum thumbnail nikl kar train kar lena ..okay ?"
"""Extract a mid-clip still (PNG) from each video in data/video_clips/."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "video_clips"
THUMBS = ROOT / "data" / "thumbnails"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def list_clips(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file())


def _probe_duration_seconds(ffmpeg: str, clip: Path) -> float | None:
    probe = Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe"))
    if probe.exists():
        proc = subprocess.run(
            [
                str(probe),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(clip),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return None
    return None


def extract_thumbnail(clip: Path, out: Path, *, at_ratio: float = 0.35) -> Path:
    """Write a single PNG frame near `at_ratio` of the clip duration (default ~35%)."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration_seconds(ffmpeg, clip)
    ss = max(0.0, (duration * at_ratio) if duration and duration > 0 else 1.0)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{ss:.3f}",
        "-i",
        str(clip),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        cmd_fb = [
            ffmpeg,
            "-y",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
        proc_fb = subprocess.run(cmd_fb, capture_output=True, text=True, check=False)
        if proc_fb.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"ffmpeg failed for {clip.name}: {(proc_fb.stderr or proc.stderr)[-500:]}"
            )
    return out


def extract_all(clips_dir: Path = CLIPS, thumbs_dir: Path = THUMBS) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for clip in list_clips(clips_dir):
        thumb = thumbs_dir / f"{clip.stem}.png"
        extract_thumbnail(clip, thumb)
        results.append(
            {
                "clip": str(clip.relative_to(ROOT)).replace("\\", "/"),
                "thumbnail": str(thumb.relative_to(ROOT)).replace("\\", "/"),
                "bytes": thumb.stat().st_size,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=CLIPS)
    parser.add_argument("--out", type=Path, default=THUMBS)
    args = parser.parse_args()
    rows = extract_all(args.clips, args.out)
    print(json.dumps({"n": len(rows), "thumbnails": rows}, indent=2))
    if not rows:
        print(
            f"No videos in {args.clips}. Drop .mp4/.mov/.mkv/.webm there, then re-run.",
            flush=True,
        )


if __name__ == "__main__":
    main()
