import json
from pathlib import Path
from faster_whisper import WhisperModel


class WhisperTranscriber:
    def __init__(self, output_dir: Path, model_name: str = "base"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # device="cpu" — для твоего Mac без GPU
        self.model = WhisperModel(model_name, device="cpu")

    def transcribe(self, audio_path: Path) -> dict:
        segments, info = self.model.transcribe(str(audio_path))

        # Собираем весь текст
        text = " ".join([seg.text for seg in segments])

        result = {
            "text": text,
            "duration": info.duration,
            "language": info.language,
        }

        output_path = self.output_dir / f"{audio_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result
