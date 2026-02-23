import json
from pathlib import Path
from typing import Dict, List


class TranscriptProcessor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, transcript: Dict, source_name: str) -> Dict:
        segments = self._extract_segments(transcript)

        full_text = " ".join(seg["text"] for seg in segments)

        processed = {
            "language": transcript.get("language"),
            "duration": transcript.get("duration"),
            "segments": segments,
            "full_text": full_text
        }

        self._save_json(processed, source_name)
        self._save_text(full_text, source_name)

        return processed

    def _extract_segments(self, transcript: Dict) -> List[Dict]:
        if "segments" in transcript and transcript["segments"]:
            cleaned = []
            for seg in transcript["segments"]:
                text = seg.get("text", "").strip()
                if not text:
                    continue

                cleaned.append({
                    "start": round(seg.get("start", 0.0), 2),
                    "end": round(seg.get("end", 0.0), 2),
                    "text": text
                })
            return cleaned

        # Fallback — Whisper returned only text without segments
        text = transcript.get("text", "").strip()
        if not text:
            return []

        duration = transcript.get("duration", 0.0)

        return [{
            "start": 0.0,
            "end": round(duration, 2),
            "text": text
        }]

    def _save_json(self, data: Dict, name: str):
        path = self.output_dir / f"{name}_processed.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_text(self, text: str, name: str):
        path = self.output_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)