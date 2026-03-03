import json
import io
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List


class TranscriptProcessor:
    """
    Handles transcription data processing:
    - Converts raw Whisper output to structured JSON and TXT.
    - Generates multi-language Excel reports.
    - Packages audit data (Excel + Video Frames) into ZIP archives.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, transcript: Dict, source_name: str) -> Dict:
        """
        Main entry point to process a raw transcript.
        Saves both JSON (for the app) and TXT (for quick review).
        """
        segments = self._extract_segments(transcript)
        full_text = " ".join(seg["text"] for seg in segments)

        processed = {
            "language": transcript.get("language"),
            "duration": transcript.get("duration"),
            "segments": segments,
            "full_text": full_text,
            "translations": transcript.get("translations", {}),
            "use_vision_flag": transcript.get("use_vision_flag", True)
        }

        self._save_json(processed, source_name)
        self._save_text(full_text, source_name)

        return processed

    def prepare_export_package(self, source_id: str, processed_data: Dict, frame_dir: Path) -> bytes:
        """
        Generates a professional Audit Export Package (ZIP).
        Contains:
        1. Excel file with synced timestamps and all available translations.
        2. 'frames' directory with visual evidence (JPGs).
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_f:
            # --- 1. Excel Generation ---
            rows = []
            segments = processed_data.get("segments", [])
            translations = processed_data.get("translations", {})

            for seg in segments:
                # Format timestamp to MM:SS for the auditor's convenience
                timestamp = f"{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}"
                row = {
                    "Timestamp": timestamp,
                    "Original_Text": seg['text']
                }

                # Align translations with original segments by start time
                for lang, t_segs in translations.items():
                    match = next((ts for ts in t_segs if abs(ts['start'] - seg['start']) < 0.1), None)
                    if match:
                        row[f"Translation_{lang}"] = match['text']

                rows.append(row)

            # Create a DataFrame and write to an Excel buffer using XlsxWriter
            df = pd.DataFrame(rows)
            excel_io = io.BytesIO()
            with pd.ExcelWriter(excel_io, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Transcript_Report')

            # Save the Excel file into the root of the ZIP
            zip_f.writestr(f"Audit_Data_{source_id}.xlsx", excel_io.getvalue())

            # --- 2. Frames Collection ---
            # Add all extracted visual frames as evidence in a sub-directory
            if frame_dir.exists():
                for img in frame_dir.glob("*.jpg"):
                    zip_f.write(img, arcname=f"frames/{img.name}")

        return buffer.getvalue()

    def _extract_segments(self, transcript: Dict) -> List[Dict]:
        """
        Extracts and cleans segments from raw Whisper/API output.
        Normalizes timestamps and removes empty entries.
        """
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

        # Fallback for transcripts without segment data
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
        """Saves processed transcript as JSON for RAG and UI consumption."""
        path = self.output_dir / f"{name}_processed.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_text(self, text: str, name: str):
        """Saves raw transcript as TXT for quick text processing/reading."""
        path = self.output_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)