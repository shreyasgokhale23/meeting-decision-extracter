#!/usr/bin/env python3
"""Manual end-to-end diarization test.

Runs the REAL diart pipeline against ``ai/data/dummy_meeting.wav`` and prints
all detected speaker turns to stdout.

Usage (from the ``ai/`` directory):
    python test_manual_diarization.py

Expected output (numbers will vary):
    [DiartProvider] Diarization complete
    ─────────────────────────────────────────
    SPEAKER_00 |   0.00s -   3.42s
    SPEAKER_01 |   3.42s -   7.15s
    ...
    ─────────────────────────────────────────
    Total segments  : 49
    Unique speakers : 3  → ['speaker0', 'speaker1', 'speaker2']
"""

import logging
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── make sure we can import from the ai/ package root ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

from diarization.diart_provider import DiartProvider  # noqa: E402

_AUDIO_PATH = os.path.join(_HERE, "data", "dummy_meeting.wav")

# ── sanity-check the WAV ─────────────────────────────────────────────────────
if not os.path.isfile(_AUDIO_PATH):
    print(
        f"[ERROR] dummy_meeting.wav not found at: {_AUDIO_PATH}\n"
        "Please ensure the file exists before running this test.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"\n[INFO] Audio file : {_AUDIO_PATH}")
print(f"[INFO] File size  : {os.path.getsize(_AUDIO_PATH):,} bytes\n")

# ── initialise provider ──────────────────────────────────────────────────────
print("[INFO] Initialising DiartProvider (loading pyannote models)…")
try:
    provider = DiartProvider()
except ImportError as exc:
    print(f"[ERROR] Import failed – is diart installed?\n  {exc}", file=sys.stderr)
    sys.exit(1)
except RuntimeError as exc:
    print(
        f"[ERROR] Pipeline initialisation failed.\n"
        f"  {exc}\n\n"
        "If this mentions Hugging Face authentication, you must:\n"
        "  1. Create a free account at https://huggingface.co\n"
        "  2. Accept the model license at:\n"
        "       https://huggingface.co/pyannote/segmentation\n"
        "       https://huggingface.co/pyannote/embedding\n"
        "  3. Set the environment variable:  HF_TOKEN=<your_token>",
        file=sys.stderr,
    )
    sys.exit(1)

print("[INFO] Provider ready. Running diarization…\n")

# ── run diarization ──────────────────────────────────────────────────────────
try:
    segments = provider.diarize(_AUDIO_PATH)
except RuntimeError as exc:
    print(f"[ERROR] Diarization failed:\n  {exc}", file=sys.stderr)
    sys.exit(1)

# ── print results ────────────────────────────────────────────────────────────
sep = "─" * 50
print(sep)
if segments:
    for seg in segments:
        print(f"  {seg.speaker:<14} | {seg.start:7.2f}s - {seg.end:7.2f}s")
else:
    print("  (no segments detected)")
print(sep)

unique_speakers = sorted({s.speaker for s in segments})
print(f"\n  Total segments  : {len(segments)}")
print(f"  Unique speakers : {len(unique_speakers)}  → {unique_speakers}\n")

print("[INFO] Manual diarization test PASSED ✓")
sys.exit(0)
