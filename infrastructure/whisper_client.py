import json
import os
from pathlib import Path
from faster_whisper import WhisperModel
import logging

# Set logging to avoid console cluttering during transcription
logging.basicConfig(level=logging.ERROR)


class WhisperTranscriber:
    """
    Faster-Whisper implementation for Lumina.
    Handles audio-to-text conversion with built-in error handling.
    """

    def __init__(self, output_dir: Path, model_name: str = "base"):
        """
        Initializes the Whisper model.
        Uses 'float32' for CPU compatibility to avoid quantization errors.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fix for potential threading issues on macOS/Linux
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        try:
            # Using CPU device for broad compatibility across specialist laptops
            self.model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="float32",
                download_root=str(self.output_dir.parent / "models")
            )
        except Exception as e:
            print(f"⚠️ Error loading Whisper model '{model_name}': {e}")
            self.model = None

    def transcribe(self, audio_path: Path, source_id: str = None) -> dict:
        """
        Transcribes audio file to text.
        Accepts source_id to ensure naming consistency across the audit pipeline.
        """
        if not audio_path or not Path(audio_path).exists():
            print(f"❌ Audio file not found: {audio_path}")
            return {}

        if self.model is None:
            print("❌ Whisper engine not initialized.")
            return {}

        try:
            # --- Transcription Settings ---
            # vad_filter: removes silences to keep the audit report concise
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=False,
                vad_parameters=dict(min_silence_duration_ms=1000),
                word_timestamps=False
            )

            result_segments = []
            # 'segments' is a generator; iterating over it triggers the actual transcription
            for seg in segments:
                text = seg.text.strip()
                if text:
                    result_segments.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": text
                    })

            result = {
                "language": info.language,
                "duration": round(info.duration, 2),
                "segments": result_segments,
                "model": "faster-whisper-base"
            }

            # Persistence: Use source_id for filename if provided, otherwise fallback to stem
            filename = source_id if source_id else Path(audio_path).stem
            output_path = self.output_dir / f"{filename}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return result

        except Exception as e:
            print(f"❌ Transcription error for {audio_path}: {e}")
            return {}