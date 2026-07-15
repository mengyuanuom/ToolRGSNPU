"""Optional microphone-to-prompt adapter using OpenAI Whisper."""

from typing import Any, Dict

from toolrgs.registry import AUDIO_INPUTS


class WhisperRecorder:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._model = None

    def transcribe_once(self) -> str:
        try:
            import sounddevice as sd
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "Audio input requires sounddevice and openai-whisper from requirement-deploy.txt"
            ) from exc
        sample_rate = int(self.cfg.get("sample_rate", 16000))
        duration_s = float(self.cfg.get("duration_s", 4.0))
        recording = sd.rec(
            int(sample_rate * duration_s),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        if self._model is None:
            self._model = whisper.load_model(
                str(self.cfg.get("model", "small")),
                device=str(self.cfg.get("device", "cpu")),
            )
        result = self._model.transcribe(
            recording.reshape(-1),
            fp16=False,
            language=self.cfg.get("language") or None,
        )
        text = str(result.get("text", "")).strip()
        if not text:
            raise RuntimeError("Whisper returned an empty instruction")
        return text


AUDIO_INPUTS.register_module(
    WhisperRecorder,
    name="whisper",
    aliases=("openai_whisper",),
)
AUDIO_INPUT_REGISTRY = AUDIO_INPUTS.module_dict


def build_audio_input(cfg: Dict[str, Any]):
    component_type = cfg.get("type", "whisper")
    try:
        audio_class = AUDIO_INPUTS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(AUDIO_INPUTS.keys()))
        raise ValueError(
            f"Unknown audio input {component_type!r}; available: {available}"
        ) from exc
    return audio_class(cfg)
