"""Microphone audio recording package for Meeting Decision Extractor."""

from audio.microphone import (
    DEFAULT_CHANNELS,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_SAMPLE_RATE,
    EmptyRecordingError,
    MicrophoneError,
    MicrophoneRecorder,
    MicrophoneRecordingError,
    MicrophoneUnavailableError,
    NotRecordingError,
    RecordingInProgressError,
    check_microphone_available,
    get_default_input_device_info,
    get_wav_info,
    record_and_process_meeting,
    record_meeting,
)

__all__ = [
    "MicrophoneRecorder",
    "record_and_process_meeting",
    "record_meeting",
    "check_microphone_available",
    "get_default_input_device_info",
    "get_wav_info",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_RECORDINGS_DIR",
    "MicrophoneError",
    "MicrophoneUnavailableError",
    "RecordingInProgressError",
    "NotRecordingError",
    "EmptyRecordingError",
    "MicrophoneRecordingError",
]
