"""Speaker-Aware Transcript Alignment Module.

Aligns OpenAI Whisper speech-to-text transcription turns with Diart speaker
diarization segments using temporal overlap matching.

Architecture & Design:
-----------------------
Whisper STT produces timestamped text intervals: [start, end] -> text.
Diart Diarization produces timestamped speaker intervals: [start, end] -> speaker.

This module matches each Whisper turn to the speaker with the MAXIMUM TEMPORAL
OVERLAP. Overlaps across multiple Diart segments for the same speaker are
accumulated. If a Whisper turn has zero overlap with any Diart segment (e.g.,
silent gaps or boundaries), a deterministic fallback speaker (default: 'UNKNOWN')
is assigned. Whisper text and timestamps are preserved verbatim.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from diarization.interface import SpeakerSegment
from models.schemas import SpeakerTranscriptTurn
from transcription.interface import TranscriptTurn


def calculate_overlap(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> float:
    """Calculate the duration of temporal overlap between two time intervals.

    Returns:
        Overlap in seconds (>= 0.0). If intervals do not overlap or are invalid,
        returns 0.0.
    """
    if start_a >= end_a or start_b >= end_b:
        return 0.0
    overlap = min(end_a, end_b) - max(start_a, start_b)
    return max(0.0, overlap)


def align_transcript_with_speakers(
    whisper_turns: List[Union[TranscriptTurn, Dict[str, Any]]],
    diarization_segments: List[Union[SpeakerSegment, Dict[str, Any]]],
    fallback_speaker: str = "UNKNOWN",
) -> List[SpeakerTranscriptTurn]:
    """Align Whisper transcript turns with Diart speaker segments.

    Algorithm:
    1. For each Whisper turn [t_start, t_end]:
       a. Compare against all Diart speaker intervals [s_start, s_end].
       b. Compute the temporal overlap in seconds:
          overlap = max(0.0, min(t_end, s_end) - max(t_start, s_start))
       c. Accumulate overlap for each unique speaker:
          total_overlap[speaker] += overlap
       d. Select the speaker with the maximum total overlap.
          Deterministic tie-breaking uses first appearance order followed by
          alphabetical speaker label.
       e. If max overlap is 0.0 (no overlap with any Diart segment), assign
          `fallback_speaker` ('UNKNOWN').
    2. Preserve Whisper text and timestamps verbatim.
    3. Ensure output turns are returned in chronological order by `start` time.

    Args:
        whisper_turns: List of TranscriptTurn objects or dicts with 'start', 'end', 'text'.
        diarization_segments: List of SpeakerSegment objects or dicts with 'start', 'end', 'speaker'.
        fallback_speaker: Speaker label assigned when no temporal overlap is found.

    Returns:
        List of SpeakerTranscriptTurn objects with assigned speakers.
    """
    if not whisper_turns:
        return []

    # Normalize Diart segments into clean tuples: (start, end, speaker, index)
    parsed_segments = []
    for idx, seg in enumerate(diarization_segments):
        try:
            if isinstance(seg, dict):
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", 0.0))
                s_speaker = str(seg.get("speaker", fallback_speaker)).strip()
            else:
                s_start = float(getattr(seg, "start", 0.0))
                s_end = float(getattr(seg, "end", 0.0))
                s_speaker = str(getattr(seg, "speaker", fallback_speaker)).strip()

            if s_start < s_end and s_speaker:
                parsed_segments.append((s_start, s_end, s_speaker, idx))
        except (ValueError, TypeError):
            continue

    aligned_turns: List[SpeakerTranscriptTurn] = []

    for turn in whisper_turns:
        try:
            if isinstance(turn, dict):
                t_start = float(turn.get("start", 0.0))
                t_end = float(turn.get("end", 0.0))
                t_text = str(turn.get("text", "")).strip()
            else:
                t_start = float(getattr(turn, "start", 0.0))
                t_end = float(getattr(turn, "end", 0.0))
                t_text = str(getattr(turn, "text", "")).strip()
        except (ValueError, TypeError):
            continue

        # Skip turns without text or invalid time range
        if not t_text:
            continue

        # Accumulate overlap per speaker
        speaker_overlaps: Dict[str, float] = defaultdict(float)
        speaker_first_seen: Dict[str, int] = {}

        for s_start, s_end, s_speaker, seg_idx in parsed_segments:
            overlap = calculate_overlap(t_start, t_end, s_start, s_end)
            if overlap > 0.0:
                speaker_overlaps[s_speaker] += overlap
                if s_speaker not in speaker_first_seen:
                    speaker_first_seen[s_speaker] = seg_idx

        # Select speaker with maximum accumulated overlap
        if speaker_overlaps:
            best_speaker = max(
                speaker_overlaps.keys(),
                key=lambda spk: (
                    speaker_overlaps[spk],
                    -speaker_first_seen.get(spk, 0),
                    spk,
                ),
            )
            # Ensure the selected overlap is non-zero
            if speaker_overlaps[best_speaker] <= 0.0:
                best_speaker = fallback_speaker
        else:
            best_speaker = fallback_speaker

        aligned_turns.append(
            SpeakerTranscriptTurn(
                speaker=best_speaker,
                start=t_start,
                end=t_end,
                text=t_text,
            )
        )

    # Sort turns chronologically
    aligned_turns.sort(key=lambda x: (x.start, x.end))
    return aligned_turns
