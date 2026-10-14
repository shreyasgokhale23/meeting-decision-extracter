import importlib, sys, inspect

diart = importlib.import_module('diart')
print('diart version:', getattr(diart, '__version__', 'unknown'))
print('has SpeakerDiarization:', hasattr(diart, 'SpeakerDiarization'))
print('has FileAudioSource:', hasattr(diart, 'FileAudioSource'))
print('has StreamingInference:', hasattr(diart, 'StreamingInference'))
if hasattr(diart, 'SpeakerDiarization'):
    from diart import SpeakerDiarization
    print('SpeakerDiarization methods:', [m for m in dir(SpeakerDiarization) if not m.startswith("_")][:10])
if hasattr(diart, 'FileAudioSource'):
    from diart import FileAudioSource
    print('FileAudioSource attrs:', [a for a in dir(FileAudioSource) if not a.startswith("_")][:10])
if hasattr(diart, 'StreamingInference'):
    from diart import StreamingInference
    print('StreamingInference attrs:', [a for a in dir(StreamingInference) if not a.startswith("_")][:10])
