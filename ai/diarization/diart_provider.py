"""Concrete DiarizationProvider backed by the diart library (v0.9+).

Workflow
--------
audio_path
    → diart.sources.FileAudioSource(audio_path, sample_rate)
    → diart.SpeakerDiarization()
    → diart.inference.StreamingInference(pipeline, source)
    → pyannote.core.Annotation
    → List[SpeakerSegment]  (sorted chronologically)

Notes
-----
- diart 0.9.x does NOT export FileAudioSource or StreamingInference from its
  top-level package. They live in ``diart.sources`` and ``diart.inference``
  respectively.  SpeakerDiarization IS exported from ``diart`` directly.
- Instantiating SpeakerDiarization (with default config) triggers model
  downloads from Hugging Face (pyannote/segmentation and pyannote/embedding).
  If the environment has a cached copy of those models no HF token is needed
  at runtime; however the *first* run may require ``HF_TOKEN`` to be set if
  the models have not been accepted/downloaded previously.
- The provider validates the audio path *before* running inference so that a
  missing file is reported immediately rather than from inside diart internals.
"""

import logging
import os
from typing import List

from diarization.interface import DiarizationProvider, SpeakerSegment

logger = logging.getLogger(__name__)

# Default sample rate expected by pyannote / diart segmentation models.
_DEFAULT_SAMPLE_RATE = 16_000


def _apply_hf_hub_compat_patch() -> None:
    """Ensure compatibility between pyannote.audio and modern huggingface_hub.

    pyannote.audio passes `use_auth_token=...` to `hf_hub_download`, but
    huggingface_hub >= 0.24 removed `use_auth_token` in favor of `token`.
    This patch safely translates `use_auth_token` to `token`.
    """
    try:
        import huggingface_hub
        import huggingface_hub.file_download

        orig_download = huggingface_hub.file_download.hf_hub_download

        def _compat_hf_hub_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                token_val = kwargs.pop("use_auth_token")
                if "token" not in kwargs:
                    kwargs["token"] = token_val
            return orig_download(*args, **kwargs)

        huggingface_hub.file_download.hf_hub_download = _compat_hf_hub_download
        huggingface_hub.hf_hub_download = _compat_hf_hub_download

        import sys
        if "pyannote.audio.core.model" in sys.modules:
            sys.modules["pyannote.audio.core.model"].hf_hub_download = _compat_hf_hub_download
    except Exception:
        pass


def _apply_torchaudio_compat_patch() -> None:
    """Ensure compatibility between diart and modern torchaudio (>=2.1).

    Modern torchaudio removed `set_audio_backend`. Diart imports still invoke
    `torchaudio.set_audio_backend('soundfile')`. Additionally, on Windows
    prebuilt torchaudio packages where `torchaudio.io` (FFmpeg) may be omitted,
    we provide a lightweight placeholder module so diart modules import cleanly.
    """
    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *args, **kwargs: None
        import sys
        import types
        try:
            import torchaudio.io
        except (ImportError, ModuleNotFoundError):
            io_mod = types.ModuleType("torchaudio.io")
            io_mod.StreamReader = type("StreamReader", (), {})
            sys.modules["torchaudio.io"] = io_mod
    except Exception:
        pass


def _apply_torch_safe_globals() -> None:
    """Allowlist required safe classes for pyannote checkpoint unpickling.

    Starting in PyTorch 2.6 (and 2.4+), `torch.load` defaults to `weights_only=True`.
    Pre-trained pyannote models (segmentation and embedding) contain metadata and
    configuration objects from OmegaConf, PyTorch Lightning, and pyannote.audio
    that are blocked under strict weights_only mode.

    Rather than disabling PyTorch security globally (weights_only=False), we
    safely allowlist only the specific legitimate metadata and container types
    using `torch.serialization.add_safe_globals`.
    """
    try:
        import torch.serialization
        if not hasattr(torch.serialization, "add_safe_globals"):
            return

        safe_classes = []

        # Built-in primitive types/collections serialized in checkpoint
        safe_classes.extend([list, dict, set, int])

        # OmegaConf containers and nodes
        try:
            import omegaconf
            import omegaconf.base
            import omegaconf.dictconfig
            import omegaconf.listconfig
            import omegaconf.nodes

            for obj in (
                getattr(omegaconf.listconfig, "ListConfig", None),
                getattr(omegaconf.dictconfig, "DictConfig", None),
                getattr(omegaconf.base, "ContainerMetadata", None),
                getattr(omegaconf.base, "Metadata", None),
                getattr(omegaconf.nodes, "AnyNode", None),
            ):
                if obj is not None:
                    safe_classes.append(obj)
        except ImportError:
            pass

        # Typing
        try:
            import typing
            safe_classes.append(typing.Any)
        except (ImportError, AttributeError):
            pass

        # Standard collections
        try:
            import collections
            safe_classes.extend([collections.defaultdict, collections.OrderedDict])
        except (ImportError, AttributeError):
            pass

        # PyTorch internals
        try:
            import torch.torch_version
            safe_classes.append(torch.torch_version.TorchVersion)
        except (ImportError, AttributeError):
            pass

        # pyannote.audio specifications and models
        try:
            import pyannote.audio.core.model
            import pyannote.audio.core.task

            for obj in (
                getattr(pyannote.audio.core.model, "Introspection", None),
                getattr(pyannote.audio.core.task, "Problem", None),
                getattr(pyannote.audio.core.task, "Specifications", None),
                getattr(pyannote.audio.core.task, "Resolution", None),
            ):
                if obj is not None:
                    safe_classes.append(obj)
        except ImportError:
            pass

        # PyTorch Lightning callback states saved in checkpoints
        try:
            import pytorch_lightning.callbacks.early_stopping
            import pytorch_lightning.callbacks.model_checkpoint

            for obj in (
                getattr(pytorch_lightning.callbacks.early_stopping, "EarlyStopping", None),
                getattr(pytorch_lightning.callbacks.model_checkpoint, "ModelCheckpoint", None),
            ):
                if obj is not None:
                    safe_classes.append(obj)
        except ImportError:
            pass

        if safe_classes:
            torch.serialization.add_safe_globals(safe_classes)
    except Exception:
        pass


# Register safe globals and compat shims at module import time
_apply_torchaudio_compat_patch()
_apply_torch_safe_globals()



class DiartProvider(DiarizationProvider):
    """Speaker diarization provider that uses diart 0.9+.

    Instantiation loads the underlying pyannote segmentation and embedding
    models (downloaded from Hugging Face on first use). All subsequent
    ``diarize`` calls reuse the same pipeline instance.
    """

    def __init__(self) -> None:
        """Initialise the diart pipeline.

        Raises
        ------
        ImportError
            If diart (or its runtime dependencies) cannot be imported.
        RuntimeError
            If the pipeline cannot be constructed (e.g. model download
            failure, missing HF token, or CUDA/CPU device error).
        """
        _apply_torchaudio_compat_patch()
        _apply_hf_hub_compat_patch()
        _apply_torch_safe_globals()

        try:
            from diart import SpeakerDiarization  # type: ignore
            from diart.blocks import SpeakerDiarizationConfig  # type: ignore
            from diart.sources import FileAudioSource  # type: ignore
            from diart.inference import StreamingInference  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "diart is not installed or could not be imported. "
                "Install it with: pip install diart"
            ) from exc

        # Store classes for use in diarize(); construct pipeline once.
        self._FileAudioSource = FileAudioSource
        self._StreamingInference = StreamingInference

        try:
            # The default delta_new=1.0 is too permissive for the online
            # clustering threshold (cosine distance).  With TTS-generated or
            # acoustically similar voices the third speaker gets merged into
            # an existing cluster.  delta_new=0.8 reliably separates 3+
            # speakers without over-segmentation (verified via sweep testing
            # across [0.3 … 1.0]).
            config = SpeakerDiarizationConfig(delta_new=0.8)
            self._pipeline = SpeakerDiarization(config=config)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialise the SpeakerDiarization pipeline. "
                "This can happen if the pyannote models have not been downloaded "
                "or if a Hugging Face token (HF_TOKEN) is required to access "
                "pyannote/segmentation and pyannote/embedding. "
                f"Original error: {exc}"
            ) from exc

        logger.info("DiartProvider initialised successfully.")

    # ------------------------------------------------------------------
    # DiarizationProvider interface
    # ------------------------------------------------------------------

    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        """Run speaker diarization on a WAV file.

        Parameters
        ----------
        audio_path:
            Absolute or relative path to the audio file (WAV recommended).

        Returns
        -------
        List[SpeakerSegment]
            Speaker turns sorted chronologically by start time.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist or is not a file.
        RuntimeError
            If the diart inference step fails.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Starting diarization for: %s", audio_path)

        # Build a file-based audio source at the pipeline's sample rate.
        sample_rate = getattr(
            getattr(self._pipeline, "config", None), "sample_rate", _DEFAULT_SAMPLE_RATE
        )
        source = self._FileAudioSource(audio_path, sample_rate)

        # Run streaming inference – returns a pyannote.core.Annotation.
        try:
            inference = self._StreamingInference(
                self._pipeline,
                source,
                do_profile=False,
                do_plot=False,
                show_progress=True,
            )
            # __call__() streams the file and returns the accumulated Annotation.
            annotation = inference()
        except Exception as exc:
            raise RuntimeError(
                f"Diart diarization failed for '{audio_path}': {exc}"
            ) from exc

        # Convert pyannote.core.Annotation tracks → SpeakerSegment list.
        speaker_segments: List[SpeakerSegment] = []
        if annotation is not None:
            for turn, _track, speaker in annotation.itertracks(yield_label=True):
                speaker_segments.append(
                    SpeakerSegment(
                        speaker=str(speaker),
                        start=float(turn.start),
                        end=float(turn.end),
                    )
                )

        if not speaker_segments:
            logger.warning("Diart returned no speaker segments for %s", audio_path)

        # Sort chronologically (diart 0.9 already returns them ordered, but
        # sorting here guarantees the contract regardless of version).
        speaker_segments.sort(key=lambda s: s.start)
        logger.info(
            "Diarization complete: %d segments, %d unique speakers",
            len(speaker_segments),
            len({s.speaker for s in speaker_segments}),
        )
        return speaker_segments
