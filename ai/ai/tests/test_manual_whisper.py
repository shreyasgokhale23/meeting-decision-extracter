import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcription.whisper_provider import WhisperProvider

provider = WhisperProvider("base")
turns = provider.transcribe("data/dummy_meeting.wav")
for turn in turns:
    print(f"{turn.start:.2f} - {turn.end:.2f}:{turn.text}")