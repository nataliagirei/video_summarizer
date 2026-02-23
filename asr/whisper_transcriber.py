import json
from pathlib import Path
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(self, output_dir: Path, model_name: str = "base"):
        """
        Initialize the Whisper transcriber with output directory and model size.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Using compute_type="float32" for CPU execution stability
        # to avoid warnings or errors on systems without NVIDIA GPUs
        self.model = WhisperModel(model_name, device="cpu", compute_type="float32")

    def transcribe(self, audio_path: Path) -> dict:
        """
        Transcribe audio file using Faster-Whisper with VAD filtering
        and high beam size for maximum precision.
        """
        # vad_filter=True removes silence to prevent model hallucinations
        # beam_size=5 increases accuracy (critical for technical audit terminology)
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        result_segments = []
        # Iterating through generator to process detected speech segments
        for seg in segments:
            text = seg.text.strip()
            if text:
                result_segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text
                })

        # Structured transcription output
        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": result_segments
        }

        # Saving transcription to JSON for traceability and audit trails
        output_path = self.output_dir / f"{audio_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result