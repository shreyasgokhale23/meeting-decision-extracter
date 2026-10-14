"""Manual verification test for Whisper transcription.

Processes ai/data/dummy_meeting.wav using the WhisperProvider and
prints the detected timestamped turns (start, end, text).
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure ai directory is on path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from transcription.whisper_provider import WhisperProvider


def main():
    audio_path = "data/dummy_meeting.wav"
    print(f"[INFO] Initialising WhisperProvider (model='base')...")
    provider = WhisperProvider("base")

    print(f"[INFO] Transcribing audio: {audio_path}...")
    turns = provider.transcribe(audio_path)

    print(f"\n[INFO] Transcription complete: {len(turns)} turns detected.")
    print("-" * 60)
    for i, turn in enumerate(turns, 1):
        print(f"[{i:02d}] {turn.start:6.2f}s - {turn.end:6.2f}s | {turn.text}")
    print("-" * 60)

    assert len(turns) > 0, "No turns returned by Whisper"
    print("\n[INFO] Manual Whisper test PASSED")


if __name__ == "__main__":
    main()
