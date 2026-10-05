"""Meeting Decision Extractor — AI/NLP Module CLI Entrypoint.

Demonstrates:
1. Real-time segment-by-segment streaming simulation via Universal Adapters.
2. Full-meeting batch processing and meeting-level intelligence extraction.
3. Clean logging and structured JSON formatting.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from models.schemas import TranscriptSegmentInput
from nlp.adapters import from_dict, from_text_line
from nlp.pipeline import process_segment, process_transcript

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai_pipeline")


def demo_real_time_streaming():
    """Simulate real-time streaming segments processed one-by-one."""
    logger.info("=== STEP 1: Real-Time Segment Streaming Demo ===")
    
    streamed_turns = [
        "[10:00:00] Rahul: Can everyone hear me?",
        "[10:00:15] Sneha: Yes, loud and clear.",
        "[10:00:45] Rahul: I'll complete the backend by Friday.",
        "[10:01:20] Sneha: Let's use Redis for caching.",
        "[10:01:55] Manager: Rahul should complete the backend by Friday.",
        "[10:02:30] Sneha: We discussed the API response time.",
        "[10:03:00] Priya: For reference, the current version is 2.1.",
        "[10:03:30] Amit: Sneha will prepare the dashboard UI tomorrow.",
    ]

    ref_date = datetime(2026, 10, 5, 10, 0, 0)

    for line in streamed_turns:
        # Convert incoming live line into standardized TranscriptSegmentInput via adapter
        input_segment = from_text_line(line, default_meeting_id="meeting_live_001")
        
        # Process single turn without needing entire transcript in memory
        output = process_segment(input_segment, reference_datetime=ref_date)
        
        logger.info(
            f"Processed turn | Type: {output.type:<11} | Person: {str(output.responsible_person):<7} | "
            f"Deadline: {str(output.deadline_normalized or output.deadline_text):<10} | "
            f"Action/Dec: {output.action or output.decision or '-'}"
        )


def demo_full_meeting_batch():
    """Demonstrate batch processing of complete meeting transcript."""
    logger.info("\n=== STEP 2: Full-Meeting Batch Processing Demo ===")

    sample_json_path = Path("data/sample_transcript.json")
    if not sample_json_path.exists():
        logger.warning(f"File {sample_json_path} not found.")
        return

    with open(sample_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Ingest through universal dictionary adapter
    segments = [from_dict(item) for item in raw_data]

    ref_date = datetime(2026, 10, 5, 10, 0, 0)
    meeting_output = process_transcript(
        segments=segments,
        meeting_id="meeting_001",
        reference_datetime=ref_date,
    )

    logger.info(f"Processed Meeting ID: {meeting_output.meeting_id}")
    logger.info(f"Participants Detected: {', '.join(meeting_output.participants)}")
    logger.info(f"Extracted Action Items: {len(meeting_output.action_items)}")
    logger.info(f"Extracted Decisions:    {len(meeting_output.decisions)}")

    print("\n--- Structured Meeting Output JSON ---")
    print(json.dumps(meeting_output.model_dump(), indent=2))


if __name__ == "__main__":
    demo_real_time_streaming()
    demo_full_meeting_batch()
