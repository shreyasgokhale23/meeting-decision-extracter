#!/usr/bin/env python3
"""Manual Verification Script for Microphone Recording and End-to-End Audio Pipeline.

Workflow:
1. Print instructions and hardware detection info.
2. Start microphone recording (16 kHz mono 16-bit PCM).
3. Allow the user to speak for an arbitrary duration.
4. Allow the user to stop recording cleanly (Press Enter or Ctrl+C).
5. Save the complete meeting to a WAV file in ai/data/recordings/.
6. Print WAV metadata (duration, sample rate, channels, bit depth, format).
7. Automatically pass the saved WAV into the existing process_audio() pipeline.
8. Print a concise summary of:
   - transcript turns (with speaker and timestamps)
   - detected participants
   - extracted action items (assignees, deadlines, statuses)
   - extracted decisions
9. Exit cleanly.

Usage:
    python test_manual_microphone.py
    python test_manual_microphone.py --duration 5   # Optional fixed duration mode
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure ai directory is on sys.path
_AI_DIR = Path(__file__).resolve().parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from audio.microphone import (
    MicrophoneRecorder,
    check_microphone_available,
    get_default_input_device_info,
    get_wav_info,
)
from transcription.audio_processor import process_audio


def main():
    parser = argparse.ArgumentParser(
        description="Manual microphone recording and meeting decision extraction test."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional recording duration in seconds. If omitted, press [ENTER] to stop.",
    )
    parser.add_argument(
        "--meeting-id",
        type=str,
        default="manual_mic_001",
        help="Meeting ID identifier (default: manual_mic_001).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit destination path for the saved WAV file.",
    )
    args = parser.parse_args()

    sep = "=" * 65
    thin_sep = "-" * 65

    print(sep)
    print("  MEETING DECISION EXTRACTOR — MANUAL MICROPHONE TEST")
    print(sep)

    # 1. Hardware Check
    if not check_microphone_available():
        print("\n[ERROR] No microphone or audio input device detected.")
        print("Please check your microphone connection and Windows audio permissions.")
        sys.exit(1)

    device_info = get_default_input_device_info()
    device_name = device_info.get("name", "Default System Microphone") if device_info else "Default Microphone"
    print(f"\n[HARDWARE] Audio Input Device: {device_name}")
    print(f"[CONFIG]   Format: 16,000 Hz, Mono (1 channel), 16-bit PCM WAV")
    print(f"[CONFIG]   Meeting ID: {args.meeting_id}")

    # 2. Instructions
    print(f"\n{thin_sep}")
    print("INSTRUCTIONS:")
    print("1. Speak naturally into your microphone.")
    print("2. You can test actions and decisions, for example:")
    print('   - "I will finish the database schema by Friday."')
    print('   - "Let\'s use Redis for caching and session management."')
    print('   - "Sneha will prepare the client presentation tomorrow."')
    if args.duration:
        print(f"3. Recording will automatically run for {args.duration:.1f} seconds.")
    else:
        print("3. Press [ENTER] whenever you are done to stop recording.")
    print(thin_sep)

    # Confirm start if interactive
    if not args.duration:
        try:
            input("\n>>> Press [ENTER] to START recording <<<")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted before recording started.")
            sys.exit(0)

    # 3. Start Recording
    recorder = MicrophoneRecorder(
        sample_rate=16000,
        channels=1,
        dtype="int16",
        recordings_dir="data/recordings",
    )

    print("\n" + sep)
    print("● RECORDING IN PROGRESS — Speak into your microphone now...")
    print(sep)

    recorder.start(meeting_id=args.meeting_id)

    try:
        if args.duration:
            print(f"Recording for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("\n>>> Press [ENTER] to STOP recording and begin AI processing <<<\n")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                print("\nStop signal received.")
    except Exception as exc:
        print(f"\n[WARNING] Recording interrupted: {exc}")
    finally:
        # 4. Stop Recording & Save WAV
        print("\nStopping audio capture and finalizing WAV file...")
        saved_wav = recorder.stop(output_path=args.output)

    # 5. Show WAV Metadata
    wav_meta = get_wav_info(saved_wav)
    print("\n" + sep)
    print("SAVED WAV RECORDING DETAILS")
    print(sep)
    print(f"  File Path      : {wav_meta['file_path']}")
    print(f"  Duration       : {wav_meta['duration_seconds']:.2f} seconds")
    print(f"  Sample Rate    : {wav_meta['sample_rate']} Hz")
    print(f"  Channels       : {wav_meta['channels']} ({'Mono' if wav_meta['channels'] == 1 else 'Stereo'})")
    print(f"  Sample Width   : {wav_meta['sample_width_bytes']} bytes ({wav_meta['bit_depth']}-bit)")
    print(f"  Format         : {wav_meta['pcm_format']}")
    print(f"  File Size      : {saved_wav.stat().st_size / 1024:.1f} KB")

    # 6. Automatic AI Pipeline Processing
    print("\n" + sep)
    print("RUNNING EXISTING AI/NLP PIPELINE")
    print("Diart Diarization -> Whisper Transcription -> Speaker Alignment -> NLP")
    print(sep)

    ref_date = datetime.now()
    try:
        result = process_audio(
            audio_path=str(saved_wav),
            meeting_id=args.meeting_id,
            reference_datetime=ref_date,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline processing failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # 7. Print Results Summary
    print("\n" + sep)
    print("MEETING INTELLIGENCE EXTRACTION RESULTS")
    print(sep)

    # Speaker Transcript
    print(f"\n--- SPEAKER-AWARE TRANSCRIPT ({len(result.speaker_transcript)} turns) ---")
    if result.speaker_transcript:
        for idx, turn in enumerate(result.speaker_transcript, 1):
            m_s = int(turn.start // 60)
            s_s = int(turn.start % 60)
            m_e = int(turn.end // 60)
            s_e = int(turn.end % 60)
            ts = f"[{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}]"
            print(f"  {idx:02d}. {ts} {turn.speaker}: \"{turn.text}\"")
    else:
        print("  (No audible speech turns detected)")

    # Action Items
    print(f"\n--- ACTION ITEMS ({len(result.action_items)}) ---")
    if result.action_items:
        for idx, action in enumerate(result.action_items, 1):
            print(f"  {idx}. Action      : {action.action}")
            print(f"     Responsible : {action.responsible_person or '-'}")
            print(f"     Deadline    : {action.deadline or '-'}")
            print(f"     Status      : {action.status}")
    else:
        print("  (No action items extracted)")

    # Decisions
    print(f"\n--- DECISIONS ({len(result.decisions)}) ---")
    if result.decisions:
        for idx, dec in enumerate(result.decisions, 1):
            print(f"  {idx}. Decision    : {dec.decision}")
            print(f"     Speaker     : {dec.speaker or '-'}")
    else:
        print("  (No decisions extracted)")

    # Summary Metrics
    print("\n" + sep)
    print("FINAL SUMMARY METRICS")
    print(sep)
    print(f"  Transcript Turns     : {len(result.speaker_transcript)}")
    print(f"  Meeting Participants : {sorted(result.participants)}")
    print(f"  Extracted Actions    : {len(result.action_items)}")
    print(f"  Extracted Decisions  : {len(result.decisions)}")
    print(sep)
    print("\n[SUCCESS] Manual microphone test finished successfully.\n")


if __name__ == "__main__":
    main()
