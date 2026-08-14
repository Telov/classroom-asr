"""Lossless Matroska audio-track discovery and model-input extraction."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioTrack:
    """One audio stream in its stable container order."""

    audio_index: int
    stream_index: int
    title: str | None
    language: str | None
    codec: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None


def probe_audio_tracks(path, *, ffprobe: str = "ffprobe") -> list[AudioTrack]:
    """Read Matroska audio-stream metadata without decoding or modifying the source."""
    source = Path(path)
    if source.suffix.casefold() not in {".mka", ".mkv"}:
        raise ValueError("expected a Matroska .mka or .mkv recording")
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,sample_rate,channels,channel_layout:"
            "stream_tags=title,language",
            "-of", "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    streams = json.loads(proc.stdout).get("streams", [])
    out = []
    for stream in streams:
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags") or {}
        rate = stream.get("sample_rate")
        out.append(
            AudioTrack(
                audio_index=len(out),
                stream_index=int(stream["index"]),
                title=tags.get("title"),
                language=tags.get("language"),
                codec=stream.get("codec_name"),
                sample_rate=int(rate) if rate else None,
                channels=int(stream["channels"]) if stream.get("channels") else None,
                channel_layout=stream.get("channel_layout"),
            )
        )
    if not out:
        raise ValueError(f"no audio streams found in {source}")
    return out


def track_by_title(tracks, title: str) -> AudioTrack:
    """Resolve a semantic track by its explicit title; never guess from ordering."""
    wanted = title.strip().casefold()
    matches = [t for t in tracks if t.title and t.title.strip().casefold() == wanted]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one audio track titled {title!r}; found {len(matches)}")
    return matches[0]


def extract_model_audio(
    source,
    track: AudioTrack,
    destination,
    *,
    ffmpeg: str = "ffmpeg",
    sampling_rate: int = 16_000,
) -> Path:
    """Decode one track to mono 16-kHz FLAC while leaving the Matroska source untouched."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-nostdin", "-v", "error", "-y",
            "-i", str(Path(source)),
            "-map", f"0:{track.stream_index}",
            "-vn", "-ac", "1", "-ar", str(sampling_rate),
            "-c:a", "flac",
            str(dest),
        ],
        check=True,
    )
    return dest
