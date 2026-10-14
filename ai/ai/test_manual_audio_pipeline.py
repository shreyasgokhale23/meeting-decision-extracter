"""End-to-End Manual Audio Pipeline Test.

Executes the full pipeline against real audio (ai/data/dummy_meeting.wav):
WAV -> Diart (3 speakers) -> Whisper (STT) -> Speaker Alignment -> NLP Pipeline
"""

import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure ai directory is on sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from transcription.audio_processor import process_audio


def main():
    audio_file = "data/dummy_meeting.wav"
    print("=" * 60)
    print("STARTING END-TO-END AUDIO PIPELINE TEST")
    print(f"Target Audio: {audio_file}")
    print("=" * 60)

    # Base reference date: Monday, October 5, 2026
    ref_date = datetime(2026, 10, 5, 10, 0, 0)

    # Process real audio through complete pipeline
    result = process_audio(
        audio_path=audio_file,
        meeting_id="meeting_dummy_001",
        reference_datetime=ref_date,
    )

    print("\n" + "=" * 60)
    print("MEETING ANALYSIS")
    print("=" * 60)

    # 1. Speaker-Aware Transcript
    print("\nSPEAKER-AWARE TRANSCRIPT\n")
    for turn in result.speaker_transcript:
        start_m = int(turn.start // 60)
        start_s = int(turn.start % 60)
        end_m = int(turn.end // 60)
        end_s = int(turn.end % 60)
        ts_str = f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]"
        print(f"{ts_str} {turn.speaker}:")
        print(f"  \"{turn.text}\"")

    # 2. Action Items
    print("\n" + "-" * 60)
    print("ACTION ITEMS\n")
    if result.action_items:
        for idx, item in enumerate(result.action_items, 1):
            # Find matching segment for speaker and timestamp details
            matching_turn = next(
                (t for t in result.meeting_summary.transcript if t.action == item.action),
                None,
            )
            speaker_label = matching_turn.speaker if matching_turn else "-"
            timestamp_str = matching_turn.timestamp if matching_turn else "-"

            print(f"{idx}.")
            print(f"  Action      : {item.action}")
            print(f"  Responsible : {item.responsible_person or '-'}")
            print(f"  Speaker     : {speaker_label}")
            print(f"  Deadline    : {item.deadline or '-'}")
            print(f"  Status      : {item.status}")
            print(f"  Timestamp   : {timestamp_str}")
    else:
        print("  No action items detected.")

    # 3. Decisions
    print("\n" + "-" * 60)
    print("DECISIONS\n")
    if result.decisions:
        for idx, item in enumerate(result.decisions, 1):
            matching_turn = next(
                (t for t in result.meeting_summary.transcript if t.decision == item.decision),
                None,
            )
            timestamp_str = matching_turn.timestamp if matching_turn else "-"

            print(f"{idx}.")
            print(f"  Decision    : {item.decision}")
            print(f"  Speaker     : {item.speaker or '-'}")
            print(f"  Timestamp   : {timestamp_str}")
    else:
        print("  No decisions detected.")

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY METRICS")
    print("=" * 60)
    unique_speakers = {t.speaker for t in result.speaker_transcript}
    print(f"Total Transcript Turns   : {len(result.speaker_transcript)}")
    print(f"Unique Speakers Detected : {len(unique_speakers)} -> {sorted(unique_speakers)}")
    print(f"Extracted Action Items   : {len(result.action_items)}")
    print(f"Extracted Decisions      : {len(result.decisions)}")
    print(f"Meeting Participants     : {sorted(result.participants)}")

    # Verification checks
    assert len(unique_speakers) >= 3, f"Expected 3 speakers, found: {unique_speakers}"
    assert len(result.speaker_transcript) > 0, "No transcript turns generated"
    assert len(result.action_items) > 0, "No action items extracted"
    assert len(result.decisions) > 0, "No decisions extracted"

    print("\n[INFO] End-to-End Audio Pipeline test PASSED ✓")


if __name__ == "__main__":
    main()
