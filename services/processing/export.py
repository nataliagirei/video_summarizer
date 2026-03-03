import io
import json
import zipfile
from pathlib import Path
from datetime import datetime
import pandas as pd

# Internal module imports
from services.processing.transcript_processor import TranscriptProcessor
from services.vision.visual_linker import VisualLinker
from infrastructure.utils.utils import format_timestamp
# Import the reporter from its correct package location
from services.processing.export_pdf import PDFReporter

class Exporter:
    """
    Service responsible for exporting processed data into various formats:
    PDF reports, Excel transcripts, and ZIP packages containing video frames.
    """

    def __init__(self, data_dirs: dict, font_dir: str | Path, insight_engine=None):
        """
        Initializes the Exporter with necessary directories and engines.

        Args:
            data_dirs (dict): Dictionary of paths (raw, processed, reports, etc.)
            font_dir (str | Path): Path to the directory containing .ttf fonts.
            insight_engine (optional): LLM/Translation engine for segment processing.
        """
        self.data_dirs = {k: Path(v) for k, v in data_dirs.items()}
        self.font_dir = Path(font_dir)
        self.insight_engine = insight_engine

        # Ensure all required export directories exist on the file system
        for key, dir_path in self.data_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)

    # ---------------- PDF EXPORT ----------------
    def generate_pdf(self, source_title: str, html_content: str, user: str = "Natalia") -> Path:
        """
        Generates a professional PDF report using the PDFReporter service.

        Returns:
            Path: The file path to the generated PDF document.
        """
        # Using the unified PDFReporter which now correctly accepts 2 arguments
        reporter = PDFReporter(self.font_dir, self.data_dirs["reports"])
        pdf_path = reporter.generate_pdf_report(source_title, html_content, user)
        return pdf_path

    # ---------------- EXCEL + FRAMES ZIP ----------------
    def create_data_package(self, source_id: str) -> bytes:
        """
        Aggregates transcript data and video frames into a single ZIP archive.
        The archive includes:
            - transcript_data.xlsx (Timestamps, Source text, and Translations)
            - A 'frames/' folder containing extracted JPG images.
        """
        processed_file = self.data_dirs["processed"] / f"{source_id}_processed.json"
        frame_dir = self.data_dirs["frames"] / source_id

        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data file missing for ID: {source_id}")

        with processed_file.open("r", encoding="utf-8") as f:
            processed_data = json.load(f)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_file:
            # --- Generate Excel spreadsheet in-memory ---
            export_rows = []
            segments = processed_data.get("segments", [])
            translations = processed_data.get("translations", {})

            for seg in segments:
                row = {
                    "Timestamp": format_timestamp(seg["start"]),
                    "Source_Text": seg["text"]
                }
                # Align translations with original segments by timestamp
                for lang_code, trans_list in translations.items():
                    match = next(
                        (t for t in trans_list if abs(t["start"] - seg["start"]) < 0.1),
                        None
                    )
                    if match:
                        row[f"Text_{lang_code}"] = match["text"]
                export_rows.append(row)

            df = pd.DataFrame(export_rows)
            excel_buffer = io.BytesIO()
            # XlsxWriter is used for better compatibility with Excel formatting
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Transcript_Data")

            zip_file.writestr("transcript_data.xlsx", excel_buffer.getvalue())

            # --- Include extracted video frames ---
            if frame_dir.exists():
                for img_path in sorted(frame_dir.glob("*.jpg")):
                    # Store frames in a sub-directory inside the ZIP
                    zip_file.write(img_path, arcname=f"frames/{img_path.name}")

        buffer.seek(0)
        return buffer.getvalue()

    # ---------------- TRANSLATIONS ----------------
    def translate_segments(self, source_id: str, target_lang: str) -> list[dict]:
        """
        Translates all transcript segments and updates the local JSON storage.
        """
        if not self.insight_engine:
            raise ValueError("Translation aborted: Insight engine not initialized.")

        processed_file = self.data_dirs["processed"] / f"{source_id}_processed.json"
        if not processed_file.exists():
            raise FileNotFoundError(f"Cannot translate: Processed file not found for {source_id}")

        with processed_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        translated_segments = []
        for seg in data.get("segments", []):
            translated_segments.append({
                **seg,
                "text": self.insight_engine.translate_text(seg["text"], target_lang)
            })

        # Save or update the translation in the JSON data structure
        data.setdefault("translations", {})[target_lang] = translated_segments

        with processed_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return translated_segments