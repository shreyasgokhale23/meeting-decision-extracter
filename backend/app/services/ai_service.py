"""AI Processing Service layer.

Thin wrapper integrating the FastAPI backend with the existing AI/NLP processing module.
Does NOT implement or duplicate NLP, Diart, Whisper, or microphone recording logic.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Ensure sibling `ai` package directory is present on sys.path
def _ensure_ai_on_path() -> Path:
    # Allow explicit override via AI_MODULE_PATH environment variable
    env_path = os.getenv("AI_MODULE_PATH")
    if env_path:
        ai_dir = Path(env_path).resolve()
    else:
        # Expected repository structure:
        # <repo_root>/
        #   ai/
        #   backend/app/services/ai_service.py
        current_file = Path(__file__).resolve()
        repo_root = current_file.parents[3]
        ai_dir = repo_root / "ai"

    if ai_dir.is_dir() and str(ai_dir) not in sys.path:
        sys.path.insert(0, str(ai_dir))
        logger.debug("Added AI module path to sys.path: %s", ai_dir)

    return ai_dir


_ensure_ai_on_path()


def process_meeting_audio(
    audio_path: Union[str, Path],
    meeting_id: str,
    reference_datetime: Optional[datetime] = None,
    transcription_provider: Optional[Any] = None,
    diarization_provider: Optional[Any] = None,
) -> Any:
    """Delegate audio processing to the existing AI/NLP module.

    Args:
        audio_path: Path to the target complete meeting WAV file.
        meeting_id: Unique meeting identifier.
        reference_datetime: Optional reference datetime for relative deadline extraction.
        transcription_provider: Optional injected transcription provider (useful for tests).
        diarization_provider: Optional injected diarization provider (useful for tests).

    Returns:
        AudioMeetingResult: Standard Pydantic result from the AI/NLP pipeline.
    """
    _ensure_ai_on_path()
    try:
        from transcription.audio_processor import process_audio
    except ImportError as exc:
        logger.error("Failed to import process_audio from AI module: %s", exc)
        raise RuntimeError(
            f"Could not import process_audio from AI module. Ensure AI requirements are installed: {exc}"
        ) from exc

    resolved_audio_path = str(Path(audio_path).resolve())
    logger.info("Delegating audio processing for meeting '%s' at '%s'", meeting_id, resolved_audio_path)

    result = process_audio(
        audio_path=resolved_audio_path,
        meeting_id=meeting_id,
        reference_datetime=reference_datetime,
        transcription_provider=transcription_provider,
        diarization_provider=diarization_provider,
    )

    return result
