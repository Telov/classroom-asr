import json
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from classroom_asr.media import AudioTrack, extract_model_audio, probe_audio_tracks, track_by_title


def test_probe_matroska_audio_tracks_preserves_stream_indices_and_titles():
    payload = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "av1"},
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "flac",
                "sample_rate": "48000",
                "channels": 1,
                "channel_layout": "mono",
                "tags": {"title": "teacher", "language": "eng"},
            },
            {
                "index": 3,
                "codec_type": "audio",
                "codec_name": "flac",
                "sample_rate": "48000",
                "channels": 1,
                "tags": {"title": "student", "language": "rus"},
            },
        ]
    }
    with patch("classroom_asr.media.subprocess.run") as run:
        run.return_value = CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        tracks = probe_audio_tracks("lesson.mka")

    assert [(t.audio_index, t.stream_index, t.title) for t in tracks] == [
        (0, 1, "teacher"),
        (1, 3, "student"),
    ]
    assert track_by_title(tracks, " STUDENT ").stream_index == 3
    assert "-show_entries" in run.call_args.args[0]


def test_track_title_must_be_explicit_and_unique():
    track = AudioTrack(0, 1, None, None, "flac", 16_000, 1, "mono")
    with pytest.raises(ValueError, match="exactly one"):
        track_by_title([track], "teacher")


def test_extract_model_audio_maps_exact_stream_and_resamples(tmp_path):
    track = AudioTrack(1, 3, "student", "rus", "flac", 48_000, 1, "mono")
    destination = tmp_path / "student.flac"
    with patch("classroom_asr.media.subprocess.run") as run:
        assert extract_model_audio("lesson.mka", track, destination) == destination

    command = run.call_args.args[0]
    assert command[command.index("-map") + 1] == "0:3"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[-1] == str(destination)
