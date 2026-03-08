import gc
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

# Force environment stability for FAISS on macOS and prevent thread conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# --- INFRASTRUCTURE IMPORTS ---
from infrastructure.youtube_client import YouTubeDownloader
from infrastructure.audio_extractor import AudioExtractor
from infrastructure.microphone_recorder import AudioRecorder
from infrastructure.whisper_client import WhisperTranscriber
from services.processing.transcript_processor import TranscriptProcessor
from services.processing.text_chunk import TextChunker
from infrastructure.vector_store import LocalVectorStore
from services.rag.rag_service import VideoInsight
from infrastructure.frame_extractor import FrameProcessor
from services.vision.visual_linker import VisualLinker
from services.processing.translation import Translator


def format_timestamp(seconds: float) -> str:
    """
    Helper function to convert float seconds into a readable MM:SS format (e.g., 05:22).
    """
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


class Pipeline:
    """
    Main orchestration class for the Lumina AI.
    Handles:
    1. Acquisition: Downloading from YouTube, recording from Mic, or processing local files.
    2. Transcription: Converting audio to text using OpenAI Whisper.
    3. RAG Indexing: Preparing data for context-aware AI chat.
    4. Visual Analysis: Extracting and linking video frames to specific text segments.
    5. Persistent Storage: Managing JSON data and file cleanup.
    """

    def __init__(self, data_dirs, whisper_model="base"):
        """
        Initializes core service engines and local storage paths.
        """
        self.DATA_DIRS = data_dirs
        self.whisper_model = whisper_model

        # Initialize technical service engines
        self.whisper_engine = WhisperTranscriber(
            self.DATA_DIRS["transcripts"],
            model_name=whisper_model
        )

        self.vector_store = LocalVectorStore()
        self.insight_engine = VideoInsight()
        self.processor = TranscriptProcessor(self.DATA_DIRS["processed"])
        self.translator = Translator()

    def run(self, source_type, source_value, duration=30, use_vision=True):
        """
        Executes the fully automated analysis pipeline.

        Args:
            source_type (str): 'YouTube', 'Microphone', or 'File'.
            source_value: The link, raw buffer, or indicator of the data source.
            duration (int): Recording length in seconds (for Microphone).
            use_vision (bool): If true, extracts frames for visual auditing.
        """
        try:
            if not source_value:
                return None

            video_path = None
            audio_path = None
            source_id = None
            display_title = "Unknown Source"

            # --- PHASE 1: DATA ACQUISITION ---
            st.info(f"Phase 1: Acquiring {source_type} data...")

            if source_type == "YouTube":
                meta = YouTubeDownloader(self.DATA_DIRS["raw"]).download(source_value)
                source_id = meta["video_id"]
                display_title = meta.get("title", f"YouTube: {source_id}")
                video_path = Path(meta["filepath"])
                audio_path = AudioExtractor(self.DATA_DIRS["audio"]).extract(video_path)
                self._register_source(source_id, display_title)

            elif source_type == "Microphone":
                audio_path = AudioRecorder(self.DATA_DIRS["audio"]).record(duration)
                source_id = audio_path.stem
                display_title = f"Mic Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                self._register_source(source_id, display_title)

            else:
                # Local File Handling
                if hasattr(source_value, "name"):
                    source_id = source_value.name.split('.')[0]
                    display_title = source_value.name
                    video_path = self.DATA_DIRS["raw"] / source_value.name

                    with open(video_path, "wb") as f:
                        f.write(source_value.getbuffer())

                    if hasattr(source_value, "type") and str(source_value.type).startswith("video"):
                        audio_path = AudioExtractor(self.DATA_DIRS["audio"]).extract(video_path)
                    else:
                        audio_path = video_path

                    self._register_source(source_id, display_title)
                else:
                    return None

            # --- PHASE 2: TRANSCRIPTION ---
            st.info("Phase 2: Transcribing audio (Whisper)...")
            transcript = self.whisper_engine.transcribe(audio_path, source_id=source_id)

            if not transcript or not transcript.get("segments"):
                st.error("Transcription failed.")
                return None

            # Initialize empty translation store and flags
            transcript["translations"] = {}
            transcript["use_vision_flag"] = use_vision
            self.processor.process(transcript, source_id)

            # --- PHASE 3: RAG INDEXING ---
            st.info("Phase 3: Indexing knowledge base for Chat...")
            try:
                chunker = TextChunker(max_tokens=150, overlap_tokens=30)
                detected_lang = transcript.get("language", "en")
                chunks = chunker.chunk_by_segments(transcript["segments"], detected_lang)

                batch_size = 5
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    prepared_batch = [
                        {
                            "id": f"{source_id}_{i + j}",
                            "text": ch.text,
                            "vector_text": getattr(ch, "vector_text", ch.text),
                            "metadata": {"source": source_id, "start": ch.start, "end": ch.end}
                        }
                        for j, ch in enumerate(batch)
                    ]
                    self.vector_store.add_chunks(prepared_batch)
                    gc.collect()

                self.vector_store.persist()
                st.success("Knowledge base updated.")
            except Exception as e:
                st.warning(f"Indexing issue: {e}")

            # --- PHASE 4: VISUAL ANALYSIS ---
            if use_vision and video_path:
                st.info("Phase 4: Extracting visual evidence...")
                try:
                    frame_dir = self.DATA_DIRS["frames"] / source_id
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    fp = FrameProcessor(frame_dir)
                    fp.extract_frames(str(video_path), interval_seconds=10)
                except Exception as ve:
                    st.warning(f"Visual analysis skipped: {ve}")

            return source_id

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return None

    def delete_source(self, source_id: str) -> bool:
        """
        Full cleanup: removes registry entries, JSON data, and extracted frames.
        """
        try:
            registry_path = self.DATA_DIRS["raw"] / "registry.json"
            if registry_path.exists():
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                if source_id in registry:
                    del registry[source_id]
                    with open(registry_path, "w", encoding="utf-8") as f:
                        json.dump(registry, f, ensure_ascii=False, indent=2)

            files_to_delete = [
                self.DATA_DIRS["transcripts"] / f"{source_id}.json",
                self.DATA_DIRS["processed"] / f"{source_id}_processed.json"
            ]
            for f in files_to_delete:
                if f.exists(): f.unlink()

            frame_dir = self.DATA_DIRS["frames"] / source_id
            if frame_dir.exists():
                for frame in frame_dir.glob("*.jpg"): frame.unlink()
                frame_dir.rmdir()
            return True
        except Exception as e:
            st.error(f"Error deleting {source_id}: {e}")
            return False

    def get_human_readable_sources(self):
        """
        Maps source IDs to user-friendly titles from registry.json for UI selectors.
        """
        REGISTRY_FILE = self.DATA_DIRS["raw"] / "registry.json"
        mapping = {}
        processed_dir = self.DATA_DIRS["processed"]
        active_ids = [f.name.replace("_processed.json", "") for f in processed_dir.glob("*_processed.json")]

        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                try:
                    registry = json.load(f)
                    for sid, meta in registry.items():
                        if sid in active_ids: mapping[meta["title"]] = sid
                except:
                    pass

        for sid in active_ids:
            if sid not in mapping.values(): mapping[f"Source: {sid}"] = sid
        return mapping

    def render_transcript_view(self, source_id: str, T: dict, ui_lang: str = None, show_original: bool = False):
        """
        Renders the interactive audit transcript with visual evidence linking.
        Allows users to toggle between the original language and the translated version.
        """
        p_file = self.DATA_DIRS["processed"] / f"{source_id}_processed.json"
        frame_dir = self.DATA_DIRS["frames"] / source_id

        if not p_file.exists():
            st.info(T["wait_msg"])
            return

        with open(p_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)

        original_segments = processed_data.get("segments", [])
        detected_lang = processed_data.get("language", "en")

        # Select data based on user toggle
        if show_original:
            segments_to_show = original_segments
        else:
            try:
                translated = self.get_translated_segments(source_id, ui_lang)
                segments_to_show = translated if translated else original_segments
            except Exception as e:
                st.warning(f"Translation error: {e}")
                segments_to_show = original_segments

        linker = VisualLinker(frame_dir, self.insight_engine)
        vision_enabled = processed_data.get("use_vision_flag", True)
        has_frames = any(frame_dir.glob("*.jpg")) if frame_dir.exists() else False

        # Link segments to frames if vision is active
        try:
            if vision_enabled and has_frames:
                anchored_data = linker.get_anchored_frames(segments_to_show, detected_lang=detected_lang)
            else:
                anchored_data = [{"start": s["start"], "audio_track": s["text"], "frame": None, "video_context": None}
                                 for s in segments_to_show]
        except:
            anchored_data = [{"start": s["start"], "audio_track": s["text"], "frame": None, "video_context": None}
                             for s in segments_to_show]

        for item in anchored_data:
            col_t, col_txt, col_media = st.columns([0.15, 0.65, 0.20])
            with col_t:
                st.caption(f"**{format_timestamp(item['start'])}**")
            with col_txt:
                st.markdown(item.get("audio_track", "..."))
                if item.get("video_context"): st.caption(item["video_context"])
            with col_media:
                if item.get("frame"): st.image(item["frame"], use_container_width=True)
            st.divider()

    def get_translated_segments(self, source_id: str, target_lang: str):
        """
        Handles translation with strict caching.
        Forces full language names (e.g. 'Polish' instead of 'PL') to avoid LLM hallucinations.
        """
        p_file = self.DATA_DIRS["processed"] / f"{source_id}_processed.json"
        if not p_file.exists(): return None

        with open(p_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "translations" not in data: data["translations"] = {}

        # 1. Return from disk if already cached for this target language
        if target_lang in data["translations"]:
            return data["translations"][target_lang]

        # 2. Map language codes to full names for the LLM
        lang_names = {"PL": "Polish", "RU": "Russian", "EN": "English"}
        target_lang_full = lang_names.get(target_lang.upper(), target_lang)

        # 3. Call the high-fidelity batch translator
        translated = self.translator.translate_segments(
            data["segments"],
            target_lang=target_lang_full
        )

        # 4. Save back to the source JSON to avoid repeated API costs
        data["translations"][target_lang] = translated
        with open(p_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return translated

    def _register_source(self, source_id, source_title):
        """Internal helper to log source metadata."""
        REGISTRY_FILE = self.DATA_DIRS["raw"] / "registry.json"
        registry = {}
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                try:
                    registry = json.load(f)
                except:
                    registry = {}
        registry[source_id] = {
            "title": source_title,
            "type": "Source",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def prepare_export(self, source_id: str):
        """Generates a ZIP bundle for export."""
        p_file = self.DATA_DIRS["processed"] / f"{source_id}_processed.json"
        frame_dir = self.DATA_DIRS["frames"] / source_id
        if not p_file.exists(): return None
        with open(p_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        return self.processor.prepare_export_package(source_id, processed_data, frame_dir)
