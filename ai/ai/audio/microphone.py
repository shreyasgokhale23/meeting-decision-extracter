"""Microphone audio recording module for Meeting Decision Extractor.

This module provides complete-meeting microphone capture and saving to a PCM WAV file,
which can then be passed to the existing AI/NLP audio processing pipeline.

Key Design Decisions:
- Complete-meeting recording first (POST-MEETING PROCESSING). No streaming or chunk-by-chunk AI processing.
- Clean separation: audio capture -> WAV file -> process_audio().
- Output format: 16 kHz mono 16-bit PCM WAV (optimal for Diart speaker diarization & Whisper STT).
- Graceful error handling for missing devices, empty audio, interrupted streams, or invalid configs.
- Reusable MicrophoneRecorder API with context manager support.
"""

import logging
import os
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# Default audio recording parameters optimal for Whisper and Diart
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"
DEFAULT_RECORDINGS_DIR = "data/recordings"


class MicrophoneError(Exception):
    """Base exception for all microphone recording operations."""


class MicrophoneUnavailableError(MicrophoneError):
    """Raised when no microphone or audio input device is available."""


class RecordingInProgressError(MicrophoneError):
    """Raised when start() is called while a recording session is already active."""


class NotRecordingError(MicrophoneError):
    """Raised when stop() is called while no recording session is active."""


class EmptyRecordingError(MicrophoneError):
    """Raised when a recording is stopped but no audio samples were captured."""


class MicrophoneRecordingError(MicrophoneError):
    """Raised when an internal audio capture or hardware streaming failure occurs."""


def check_microphone_available() -> bool:
    """Check whether at least one audio input device is available on the system.

    Returns:
        bool: True if an audio input device is accessible, False otherwise.
    """
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        # sounddevice returns either a DeviceList (list of dicts) or a single dict
        if isinstance(devices, dict):
            devices = [devices]
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception as exc:
        logger.warning("Failed to query audio input devices: %s", exc)
        return False


def get_default_input_device_info() -> Optional[Dict[str, Any]]:
    """Retrieve metadata about the default audio input device.

    Returns:
        dict: Device properties dictionary, or None if no input device is found.
    """
    try:
        import sounddevice as sd

        default_device_idx = sd.default.device[0]
        if default_device_idx == -1 or default_device_idx is None:
            return None
        info = sd.query_devices(default_device_idx)
        return dict(info) if isinstance(info, dict) else None
    except Exception as exc:
        logger.debug("Failed to get default input device info: %s", exc)
        return None


def get_wav_info(wav_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect and return metadata for a PCM WAV file.

    Args:
        wav_path: Path to the target WAV file.

    Returns:
        dict: Metadata dictionary containing duration (seconds), sample_rate,
              channels, sample_width, frame_count, and format summary.

    Raises:
        FileNotFoundError: If the WAV file does not exist.
        wave.Error: If the file is not a valid WAV format.
    """
    path_obj = Path(wav_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"WAV file not found: {path_obj}")

    with wave.open(str(path_obj), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = float(n_frames) / float(framerate) if framerate > 0 else 0.0

    return {
        "file_path": str(path_obj.resolve()),
        "duration_seconds": round(duration, 3),
        "sample_rate": framerate,
        "channels": n_channels,
        "sample_width_bytes": sampwidth,
        "bit_depth": sampwidth * 8,
        "total_frames": n_frames,
        "pcm_format": f"PCM_{sampwidth * 8}bit_{'mono' if n_channels == 1 else 'stereo'}",
    }


def _sanitize_id(identifier: str) -> str:
    """Sanitize meeting identifier to ensure safe filesystem path naming."""
    clean = re.sub(r"[^\w\-_]", "_", str(identifier)).strip("._")
    return clean or "meeting"


class MicrophoneRecorder:
    """Thread-safe microphone recorder capturing arbitrary-duration audio to WAV.

    Audio samples are accumulated in memory buffers during the recording session
    and persisted as standard PCM 16-bit WAV upon calling `stop()`.

    Features:
    - Arbitrary duration: continues until explicitly stopped.
    - Zero chunk AI processing during recording: avoids blocking or real-time latency.
    - Automatic output directory creation and collision-free timestamped naming.
    - Clean resource cleanup with Context Manager support (`with MicrophoneRecorder() ...`).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        dtype: str = DEFAULT_DTYPE,
        blocksize: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
        recordings_dir: Union[str, Path] = DEFAULT_RECORDINGS_DIR,
    ) -> None:
        """Initialize the recorder configuration.

        Args:
            sample_rate: Audio sampling frequency in Hz (default: 16000 Hz).
            channels: Number of audio channels (default: 1, mono).
            dtype: Numpy/sounddevice sample data type (default: "int16").
            blocksize: Optional block size for the audio callback (default: None for system default).
            device: Optional sounddevice device index or name (default: None for default mic).
            recordings_dir: Directory where recorded WAV files are saved (default: data/recordings).
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer, got: {sample_rate}")
        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got: {channels}")
        if dtype not in ("int16", "int32", "float32"):
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported: 'int16', 'int32', 'float32'")

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        self.recordings_dir = Path(recordings_dir)

        self._frames: List[np.ndarray] = []
        self._stream: Any = None
        self._is_recording: bool = False
        self._current_meeting_id: Optional[str] = None
        self._start_time: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        """Return True if an audio recording session is currently active."""
        return self._is_recording

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback invoked by sounddevice InputStream on each audio buffer."""
        if status:
            logger.warning("Microphone stream status warning: %s", status)
        self._frames.append(indata.copy())

    def start(self, meeting_id: Optional[str] = None) -> None:
        """Start capturing audio from the microphone.

        Args:
            meeting_id: Optional meeting identifier for naming the saved recording file.

        Raises:
            RecordingInProgressError: If a recording is already active.
            MicrophoneUnavailableError: If no microphone device is found.
            MicrophoneRecordingError: If opening or starting the audio stream fails.
        """
        if self._is_recording:
            raise RecordingInProgressError(
                "A recording session is already active. Call stop() before starting a new recording."
            )

        if not check_microphone_available():
            raise MicrophoneUnavailableError(
                "No microphone or audio input device was found on this system. "
                "Please connect a working microphone and verify microphone permissions."
            )

        try:
            import sounddevice as sd

            self._frames = []
            self._current_meeting_id = _sanitize_id(meeting_id or f"meeting_{uuid.uuid4().hex[:8]}")

            logger.info(
                "Initializing microphone stream (rate=%d, channels=%d, dtype=%s, device=%s)...",
                self.sample_rate,
                self.channels,
                self.dtype,
                self.device,
            )

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.blocksize,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            self._start_time = time.time()
            logger.info("Microphone recording started for meeting '%s'.", self._current_meeting_id)
        except MicrophoneError:
            self.cleanup()
            raise
        except Exception as exc:
            self.cleanup()
            raise MicrophoneRecordingError(
                f"Failed to start microphone recording: {exc}. "
                "Verify microphone permissions and audio driver settings."
            ) from exc

    def stop(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Stop capturing audio and write the complete meeting to a WAV file.

        Args:
            output_path: Optional explicit file path for saving the WAV.
                         If omitted, a unique timestamped file is generated
                         inside `self.recordings_dir`.

        Returns:
            Path: Resolved Path to the saved WAV file.

        Raises:
            NotRecordingError: If stop() is called when no recording is active.
            EmptyRecordingError: If the recording buffer contains zero audio samples.
            MicrophoneRecordingError: If writing the WAV file fails.
        """
        if not self._is_recording:
            raise NotRecordingError(
                "No recording is currently active. Call start() before calling stop()."
            )

        logger.info("Stopping microphone stream...")
        try:
            if self._stream is not None:
                if getattr(self._stream, "active", False):
                    self._stream.stop()
                self._stream.close()
        except Exception as exc:
            logger.warning("Error closing audio stream: %s", exc)
        finally:
            self._stream = None
            self._is_recording = False

        duration = (time.time() - self._start_time) if self._start_time else 0.0

        if not self._frames:
            raise EmptyRecordingError(
                f"Recording session ended after {duration:.2f}s with 0 audio frames captured. "
                "Ensure your microphone is active and not muted."
            )

        try:
            audio_array = np.concatenate(self._frames, axis=0)
        except Exception as exc:
            raise MicrophoneRecordingError(f"Failed to assemble recorded audio buffers: {exc}") from exc

        if len(audio_array) == 0:
            raise EmptyRecordingError(
                f"Recorded audio array is empty (0 samples captured over {duration:.2f}s)."
            )

        # Determine target file destination
        if output_path is not None:
            target_file = Path(output_path).resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            m_id = self._current_meeting_id or "meeting"
            filename = f"meeting_{m_id}_{timestamp}.wav"
            target_file = (self.recordings_dir / filename).resolve()

        # Ensure parent directory exists
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise MicrophoneRecordingError(
                f"Failed to create directory '{target_file.parent}': {exc}"
            ) from exc

        # Save to standard 16-bit PCM WAV
        try:
            # Convert sample data to 16-bit PCM bytes
            if self.dtype == "int16":
                pcm_data = audio_array.tobytes()
                sampwidth = 2
            elif self.dtype == "int32":
                # Convert int32 to int16 for universal Whisper/Diart compatibility
                pcm_data = (audio_array >> 16).astype(np.int16).tobytes()
                sampwidth = 2
            elif self.dtype == "float32":
                # Scale float32 [-1.0, 1.0] to int16
                scaled = np.clip(audio_array, -1.0, 1.0) * 32767.0
                pcm_data = scaled.astype(np.int16).tobytes()
                sampwidth = 2
            else:
                pcm_data = audio_array.tobytes()
                sampwidth = 2

            with wave.open(str(target_file), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_data)

            total_frames = len(audio_array)
            actual_duration = float(total_frames) / float(self.sample_rate)
            file_size_kb = target_file.stat().st_size / 1024.0

            logger.info(
                "Successfully saved recording to '%s' (Duration: %.2fs, Size: %.1f KB).",
                target_file,
                actual_duration,
                file_size_kb,
            )
            return target_file
        except Exception as exc:
            raise MicrophoneRecordingError(
                f"Failed to write audio recording to WAV file '{target_file}': {exc}"
            ) from exc

    def record_until_stopped(
        self,
        meeting_id: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        stop_trigger: Optional[Callable[[], Any]] = None,
    ) -> Path:
        """Start recording and block execution until stopped via trigger or user input.

        Args:
            meeting_id: Optional session identifier.
            output_path: Optional destination path for the WAV file.
            stop_trigger: Optional callback invoked to determine when to stop.
                          If None, prompts the console user: 'Press Enter to stop recording...'.

        Returns:
            Path: Path to the saved WAV recording.
        """
        self.start(meeting_id=meeting_id)
        try:
            if stop_trigger is not None:
                stop_trigger()
            else:
                print("\n[REC] Recording in progress... Speak into your microphone.")
                print(">>> Press [ENTER] when the meeting finishes to stop recording <<<\n")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    print("\n[INFO] Stop signal received.")
            return self.stop(output_path=output_path)
        except Exception:
            self.cleanup()
            raise

    def cleanup(self) -> None:
        """Safely release and terminate any active audio stream resources."""
        if self._stream is not None:
            try:
                if getattr(self._stream, "active", False):
                    self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Exception during audio stream cleanup: %s", exc)
            finally:
                self._stream = None
                self._is_recording = False

    def __enter__(self) -> "MicrophoneRecorder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()


def record_and_process_meeting(
    meeting_id: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    reference_datetime: Optional[datetime] = None,
    recorder: Optional[MicrophoneRecorder] = None,
    stop_trigger: Optional[Callable[[], Any]] = None,
    transcription_provider: Optional[Any] = None,
    diarization_provider: Optional[Any] = None,
) -> Any:
    """Orchestrate full meeting lifecycle: Record Microphone -> Save WAV -> Run process_audio().

    Workflow:
        1. Initialize and start microphone capture.
        2. Wait until recording is stopped via `stop_trigger` or user confirmation.
        3. Persist complete meeting as a valid PCM WAV file.
        4. Pass the resulting WAV path to the existing `process_audio()` pipeline.
        5. Return the verified `AudioMeetingResult`.

    Args:
        meeting_id: Unique meeting ID passed consistently into process_audio().
        output_path: Optional custom path for the saved WAV file.
        sample_rate: Sampling frequency (default: 16000 Hz).
        channels: Channel count (default: 1, mono).
        reference_datetime: Optional base date for relative deadline resolution.
        recorder: Optional custom MicrophoneRecorder instance (useful for mocking).
        stop_trigger: Optional callable executed to wait for meeting termination.
        transcription_provider: Optional custom TranscriptionProvider (e.g. for offline testing).
        diarization_provider: Optional custom DiarizationProvider (e.g. for offline testing).

    Returns:
        AudioMeetingResult: Complete structured meeting intelligence from the existing AI/NLP core.

    Raises:
        MicrophoneError: If audio capture or saving fails.
        Exception: If process_audio fails.
    """
    from transcription.audio_processor import process_audio

    m_id = meeting_id or f"meeting_{uuid.uuid4().hex[:8]}"
    rec = recorder or MicrophoneRecorder(
        sample_rate=sample_rate,
        channels=channels,
    )

    try:
        logger.info("Initiating meeting recording for meeting_id='%s'...", m_id)
        saved_wav_path = rec.record_until_stopped(
            meeting_id=m_id,
            output_path=output_path,
            stop_trigger=stop_trigger,
        )
        logger.info("Audio recording saved at: %s", saved_wav_path)
    except Exception:
        rec.cleanup()
        raise

    logger.info("Passing recorded WAV to existing process_audio() pipeline...")
    result = process_audio(
        audio_path=str(saved_wav_path),
        meeting_id=m_id,
        reference_datetime=reference_datetime,
        transcription_provider=transcription_provider,
        diarization_provider=diarization_provider,
    )

    logger.info("Meeting processing complete for meeting_id='%s'.", m_id)
    return result


# Convenience alias matching functional terminology
record_meeting = record_and_process_meeting
