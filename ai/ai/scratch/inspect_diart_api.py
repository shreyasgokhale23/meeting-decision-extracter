"""Inspect the installed diart 0.2.5 package API without running inference."""
import sys
import traceback

print("=== Python version ===")
print(sys.version)
print()

# 1. Top-level diart namespace
print("=== diart top-level imports ===")
try:
    import diart
    print("diart version:", getattr(diart, "__version__", "unknown"))
    print("dir(diart):", [a for a in dir(diart) if not a.startswith("_")])
except Exception as e:
    print("FAILED to import diart:", e)
    traceback.print_exc()
    sys.exit(1)

print()

# 2. SpeakerDiarization
print("=== SpeakerDiarization ===")
try:
    from diart import SpeakerDiarization
    print("  Import OK:", SpeakerDiarization)
    print("  Bases:", [b.__name__ for b in SpeakerDiarization.__mro__])
    print("  Methods:", [m for m in dir(SpeakerDiarization) if not m.startswith("_")])
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print()

# 3. FileAudioSource
print("=== FileAudioSource ===")
try:
    from diart import FileAudioSource
    print("  Import OK:", FileAudioSource)
    print("  Signature:", FileAudioSource.__init__.__doc__)
    import inspect
    print("  __init__ params:", list(inspect.signature(FileAudioSource.__init__).parameters.keys()))
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print()

# 4. StreamingInference
print("=== StreamingInference ===")
try:
    from diart import StreamingInference
    print("  Import OK:", StreamingInference)
    import inspect
    print("  __init__ params:", list(inspect.signature(StreamingInference.__init__).parameters.keys()))
    print("  __call__ params:", list(inspect.signature(StreamingInference.__call__).parameters.keys()) if hasattr(StreamingInference, "__call__") else "N/A")
    print("  Methods:", [m for m in dir(StreamingInference) if not m.startswith("_")])
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print()

# 5. pyannote.core.Annotation
print("=== pyannote.core.Annotation ===")
try:
    from pyannote.core import Annotation, Segment
    print("  Import OK")
    ann = Annotation()
    ann[Segment(0.0, 2.0)] = "SPEAKER_00"
    ann[Segment(2.0, 5.0)] = "SPEAKER_01"
    for turn, track, label in ann.itertracks(yield_label=True):
        print(f"  turn={turn}, track={track}, label={label}")
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print()

# 6. Check what itertracks returns signature
print("=== Annotation.itertracks signature ===")
try:
    from pyannote.core import Annotation
    import inspect
    print("  params:", list(inspect.signature(Annotation.itertracks).parameters.keys()))
except Exception as e:
    print("  FAILED:", e)

print()
print("=== Installed versions ===")
try:
    import importlib.metadata as im
    for pkg in ["diart", "torch", "torchaudio", "pyannote.audio", "pyannote.core"]:
        try:
            print(f"  {pkg}: {im.version(pkg)}")
        except Exception:
            print(f"  {pkg}: not found")
except Exception as e:
    print("  FAILED:", e)
