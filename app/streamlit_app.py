import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
from streamlit_quill import st_quill
import io
import zipfile


# --- INITIAL CONFIG & SECURITY ---
load_dotenv()

# --- MODULE IMPORTS ---
from infrastructure.youtube_client import YouTubeDownloader
from infrastructure.audio_extractor import AudioExtractor
from infrastructure.microphone_recorder import AudioRecorder
from infrastructure.whisper_client import WhisperTranscriber
from services.processing.transcript_processor import TranscriptProcessor
from services.processing.text_chunk import TextChunker
from infrastructure.vector_store import LocalVectorStore
from services.rag.rag_service import VideoInsight

# NOWE MODUŁY
from infrastructure.frame_extractor import FrameProcessor
from services.vision.visual_linker import VisualLinker

# --- DIRECTORIES CONFIGURATION ---
# Dodano "frames" do ścieżek
DATA_DIRS = {k: Path(f"data/{k}") for k in ["raw", "audio", "transcripts", "processed", "reports", "frames"]}
FONT_DIR = Path("../assets/fonts")
FONT_DIR.mkdir(parents=True, exist_ok=True)

for directory in DATA_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

DRAFTS_FILE = DATA_DIRS["raw"] / "drafts.json"
REGISTRY_FILE = DATA_DIRS["raw"] / "registry.json"


# --- UTILS & CORE LOGIC ---
def format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS format."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def prepare_content_for_editor(text: str) -> str:
    """Prepares text for Quill editor by handling line breaks."""
    return text.replace('\n', '<br>')


def save_draft(title, content):
    """Saves edited HTML content to a local JSON file."""
    drafts = load_drafts()
    drafts[title] = {
        "content": content,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    with open(DRAFTS_FILE, "w", encoding="utf-8") as f:
        json.dump(drafts, f, ensure_ascii=False, indent=2)


def load_drafts():
    """Retrieves all saved drafts."""
    if DRAFTS_FILE.exists():
        with open(DRAFTS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


@st.cache_resource
def get_whisper_model(model_name):
    return WhisperTranscriber(DATA_DIRS["transcripts"], model_name=model_name)


@st.cache_resource
def get_vector_store():
    return LocalVectorStore()


@st.cache_resource
def get_insight_engine():
    return VideoInsight()


# --- PDF REPORTER CLASS ---
class AuditReporter(FPDF):
    def __init__(self, font_dir, output_dir):
        super().__init__()
        self.font_dir = Path(font_dir)
        self.output_dir = Path(output_dir)

    def generate_pdf_report(self, source_title, content, user="Natalia"):
        self.add_page()
        fonts = {
            "reg": self.font_dir / "DejaVuSans.ttf",
            "bold": self.font_dir / "DejaVuSans-Bold.ttf",
            "ital": self.font_dir / "DejaVuSans-Oblique.ttf",
            "bi": self.font_dir / "DejaVuSans-BoldOblique.ttf",
            "kr": self.font_dir / "NotoSansKR-Regular.ttf"
        }
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in content)

        if has_korean and fonts["kr"].exists():
            self.add_font("MainFont", "", str(fonts["kr"]), uni=True)
            font_family = "MainFont"
        elif fonts["reg"].exists():
            self.add_font("MainFont", "", str(fonts["reg"]), uni=True)
            self.add_font("MainFont", "B", str(fonts["bold"]), uni=True)
            font_family = "MainFont"
        else:
            font_family = "Arial"

        self.set_font(font_family, 'B', 16)
        self.multi_cell(0, 10, f"Audit Report: {source_title}", align='C')
        self.set_font(font_family, '', 10)
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Specialist: {user}", ln=True)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(5)
        self.set_font(font_family, '', 11)
        self.write_formatted_html(content, font_family)

        report_name = f"Lumina_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = self.output_dir / report_name
        self.output(str(report_path))
        return report_path

    def write_formatted_html(self, html, font_name):
        html = html.replace('<ol>', '<ol_start>').replace('</ol>', '<ol_end>')
        html = html.replace('<ul>', '').replace('</ul>', '').replace('<p>', '').replace('</p>', '\n').replace('<br>',
                                                                                                              '\n')
        parts = re.split(r'(<[^>]+>)', html)
        is_ol, idx, style = False, 1, ""
        for part in parts:
            if not part: continue
            if part in ['<b>', '<strong>']:
                style += 'B';
                self.set_font(font_name, style)
            elif part in ['</b>', '</strong>']:
                style = style.replace('B', '');
                self.set_font(font_name, style)
            elif part == '<ol_start>':
                is_ol, idx = True, 1
            elif part == '<ol_end>':
                is_ol = False
            elif part == '<li>':
                self.write(7, f"\n {idx}. " if is_ol else "\n • ")
                if is_ol: idx += 1
            elif not part.startswith('<'):
                if self.get_y() > 270: self.add_page()
                self.write(7, part)


# --- UI TRANSLATION ---
LANG_DICT = {
    "EN": {
        "title": "Lumina", "sidebar_header": "⚙️ Management", "option": "Method:",
        "run_btn": "🚀 Process", "chat_header": "💬 Chat", "chat_placeholder": "Ask...",
        "trans_header": "📄 Transcript", "wait_msg": "👈 Add source.", "quality_option": "Model:",
        "mom_btn": "Meeting Minutes", "audit_btn": "Sentiment Analysis", "export_btn": "📥 Export PDF",
        "del_btn": "Delete", "drafts_header": "📝 Drafts", "save_btn": "💾 Save Draft",
        "open_btn": "📂 Open", "source_select": "📚 Sources:",
        "no_drafts": "No drafts found.", "choose_source": "Choose options",
        "method_file": "File", "method_mic": "Microphone",
        "quality_fast": "Fast (base)", "quality_high": "High (medium)",
        "mic_warning": "🎤 Microphone mode: Recording will start after clicking the button below.",
        "file_label": "Upload File:", "rec_duration": "Recording duration (seconds):",
        "visual_toggle": "Enable Visual Analysis (AI Vision)"
    },
    "RU": {
        "title": "Lumina", "sidebar_header": "⚙️ Управление", "option": "Метод:",
        "run_btn": "🚀 Обработать", "chat_header": "💬 Чат", "chat_placeholder": "Спросите...",
        "trans_header": "📄 Транскрипт", "wait_msg": "👈 Добавьте источник.", "quality_option": "Model:",
        "mom_btn": "Протокол совещания", "audit_btn": "Анализ тональности", "export_btn": "📥 Экспорт PDF",
        "del_btn": "Удалить", "drafts_header": "📝 Черновики", "save_btn": "💾 Сохранить",
        "open_btn": "📂 Открыть", "source_select": "📚 Источники:",
        "no_drafts": "Черновиki не найдены.", "choose_source": "Выберите источники",
        "method_file": "Файл", "method_mic": "Микрофон",
        "quality_fast": "Быстрый (base)", "quality_high": "Высокое качество (medium)",
        "mic_warning": "🎤 Режим микрофона: запись начнется после нажатия кнопки.",
        "file_label": "Загрузить файл:", "rec_duration": "Длительность записи (сек):",
        "visual_toggle": "Включить визуальный анализ (AI Vision)"
    },
    "PL": {
        "title": "Lumina", "sidebar_header": "⚙️ Zarządzanie", "option": "Metoda:",
        "run_btn": "🚀 Przetwórz", "chat_header": "💬 Czat", "chat_placeholder": "Zapytaj...",
        "trans_header": "📄 Transkrypcja", "wait_msg": "👈 Dodaj źródło.", "quality_option": "Model:",
        "mom_btn": "Minutki ze spotkania", "audit_btn": "Analiza sentymentu", "export_btn": "📥 Eksport PDF",
        "del_btn": "Usuń źródło", "drafts_header": "📝 Szkice", "save_btn": "Zapisz szkic",
        "open_btn": "📂 Otwórz", "source_select": "📚 Źródła:",
        "no_drafts": "Nie znaleziono szkiców.", "choose_source": "Wybierz źródła",
        "method_file": "Plik", "method_mic": "Mikrofon",
        "quality_fast": "Szybki (base)", "quality_high": "Wysoka jakość (medium)",
        "mic_warning": "🎤 Tryb mikrofonu: Nagrywanie rozpocznie się po kliknięciu przycisku poniżej.",
        "file_label": "Wgraj plik:", "rec_duration": "Czas nagrania (sekundy):",
        "visual_toggle": "Włącz analizę wizualną (AI Vision)"
    }
}


# --- DATA GOVERNANCE ---
def delete_source_data(source_id, source_title=None):
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)
        if source_id in registry:
            del registry[source_id]
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)

    if source_title:
        drafts = load_drafts()
        if source_title in drafts:
            del drafts[source_title]
            with open(DRAFTS_FILE, "w", encoding="utf-8") as f:
                json.dump(drafts, f, ensure_ascii=False, indent=2)

    for folder in DATA_DIRS.values():
        for file in folder.glob(f"{source_id}*"):
            try:
                file.unlink()
            except:
                pass


def get_human_readable_sources():
    mapping = {}
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)
            for sid, meta in registry.items():
                mapping[meta["title"]] = sid
    return mapping


def create_data_package(source_id, processed_data, frame_dir):
    """
    Generates a ZIP archive containing:
    1. Excel file with timestamps and all language versions.
    2. 'frames' directory with extracted JPG images.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        # --- 1. Excel Spreadsheet Generation ---
        export_data = []
        segments = processed_data.get("segments", [])
        translations = processed_data.get("translations", {})

        for seg in segments:
            entry = {
                "Timestamp": format_timestamp(seg['start']),
                "Source_Text": seg['text']
            }
            # Append available translations as separate columns
            for lang_code, trans_segs in translations.items():
                match = next((ts for ts in trans_segs if abs(ts['start'] - seg['start']) < 0.1), None)
                if match:
                    entry[f"Text_{lang_code}"] = match['text']
            export_data.append(entry)

        df = pd.DataFrame(export_data)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Transcript_Data')

        zip_file.writestr("transcript_data.xlsx", excel_buffer.getvalue())

        # --- 2. Image Asset Collection ---
        if frame_dir.exists():
            for img_path in frame_dir.glob("*.jpg"):
                zip_file.write(img_path, arcname=f"frames/{img_path.name}")

    return buffer.getvalue()


def run_pipeline(source_type, source_value, duration, model_type, use_vision=True):
    try:
        video_path = None
        if source_type == "YouTube":
            metadata = YouTubeDownloader(DATA_DIRS["raw"]).download(source_value)
            source_id = metadata["video_id"]
            video_path = Path(metadata["filepath"])
            audio_path = AudioExtractor(DATA_DIRS["audio"]).extract(video_path)
        elif source_type == "Microphone":
            audio_path = AudioRecorder(DATA_DIRS["audio"]).record(duration=duration)
            source_id = audio_path.stem
            source_title = f"Mic Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            registry = {}
            if REGISTRY_FILE.exists():
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    try:
                        registry = json.load(f)
                    except:
                        registry = {}

            registry[source_id] = {
                "title": source_title,
                "type": "Microphone",
                "date": datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
        else:
            source_id = source_value.name.split('.')[0]
            video_path = DATA_DIRS["raw"] / source_value.name
            with open(video_path, "wb") as f:
                f.write(source_value.getbuffer())
            audio_path = AudioExtractor(DATA_DIRS["audio"]).extract(video_path) if source_value.type.startswith(
                "video") else video_path

        # 1. AI Transcription (Base Language only)
        whisper_engine = get_whisper_model(model_type)
        transcript = whisper_engine.transcribe(audio_path)
        if not transcript["segments"]:
            return None

        # Initialize empty translations dictionary to save tokens
        # Translations will be performed on-demand in the UI
        transcript["translations"] = {}
        transcript["use_vision_flag"] = use_vision

        # Save the processed data (Initial state: original language only)
        TranscriptProcessor(DATA_DIRS["processed"]).process(transcript, source_id)

        # 2. Visual Analysis: Frame Extraction
        if use_vision and video_path and source_type != "Microphone":
            frame_dir = DATA_DIRS["frames"] / source_id
            frame_dir.mkdir(parents=True, exist_ok=True)

            if video_path.exists():
                fp = FrameProcessor(frame_dir)
                res = fp.extract_frames(str(video_path), interval_seconds=10)
                print(f"DEBUG: Extracted {len(res)} frames for {source_id}")
            else:
                st.error(f"Video file not found: {video_path}")
        else:
            print(f"INFO: Visual analysis skipped for {source_id}")

        # 3. Knowledge Base: Vector Store Indexing
        chunker = TextChunker(max_tokens=150, overlap_tokens=30)
        # Index original transcript for RAG
        chunks = chunker.chunk_by_segments(transcript["segments"], transcript.get("language", "en"))
        prepared_chunks = [
            {
                "id": f"{source_id}_{i}",
                "text": ch.text,
                "vector_text": ch.vector_text,
                "metadata": {"source": source_id, "start": ch.start, "end": ch.end}
            } for i, ch in enumerate(chunks)
        ]

        store = get_vector_store()
        store.add_chunks(prepared_chunks)
        store.persist()

        return source_id

    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        return None


# --- UI APP CONFIG ---
st.set_page_config(page_title="Lumina", layout="wide")
if "ui_lang" not in st.session_state: st.session_state.ui_lang = "PL"
if "last_analysis" not in st.session_state: st.session_state.last_analysis = ""
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR SECTION ---
with st.sidebar:
    # 1. Language Selection
    # Synchronizes the UI language with the session state
    st.session_state.ui_lang = st.selectbox(
        "🌐 Language",
        ["PL", "RU", "EN"],
        index=["PL", "RU", "EN"].index(st.session_state.ui_lang)
    )

    # Load translation dictionary based on selected language
    T = LANG_DICT[st.session_state.ui_lang]
    # st.header(T["sidebar_header"])

    # 2. Drafts Management
    # Allows loading previously saved audit report drafts
    st.subheader(T["drafts_header"])
    all_drafts = load_drafts()
    if all_drafts:
        sel_draft = st.selectbox("Load Draft:", options=list(all_drafts.keys()))
        if st.button(T["open_btn"]):
            st.session_state.last_analysis = all_drafts[sel_draft]["content"]
    else:
        st.info(T["no_drafts"])

    st.divider()

    # 3. Input Method Selection
    # Maps translated UI labels to internal technical keys for the pipeline
    methods_map = {
        "YouTube": "YouTube",
        T["method_file"]: "File",
        T["method_mic"]: "Microphone"
    }

    selected_method_label = st.radio(T["option"], list(methods_map.keys()))
    internal_option = methods_map[selected_method_label]

    # --- NEW FEATURE: AI Vision Toggle ---
    # This toggle allows the user to bypass frame extraction and Gemini Vision API calls.
    # Essential for saving API quotas when only text transcription is needed.
    use_vision_enabled = st.toggle(T["visual_toggle"], value=True)

    # 4. Source Value Handling
    source_val = None
    rec_duration = 30  # Default duration for microphone

    if internal_option == "YouTube":
        source_val = st.text_input("Link:")
    elif internal_option == "File":
        source_val = st.file_uploader(T["file_label"])
    elif internal_option == "Microphone":
        st.warning(T["mic_warning"])
        rec_duration = st.slider(T["rec_duration"], 5, 120, 30)
        source_val = "MIC_ACTIVE"

    # 5. Model Quality Selection
    # Maps user-friendly labels to Whisper model sizes (base vs medium)
    quality_map = {
        T["quality_fast"]: "base",
        T["quality_high"]: "medium"
    }
    selected_quality_label = st.radio(T["quality_option"], list(quality_map.keys()))
    actual_model_type = quality_map[selected_quality_label]

    # 6. Execution Trigger
    # Runs the services pipeline with selected parameters
    if st.button(T["run_btn"], use_container_width=True):
        # Determine specific settings for different source types
        if internal_option == "Microphone":
            with st.spinner(f"🔴 {T['run_btn']}..."):
                # Microphone recordings never use visual analysis
                res_id = run_pipeline(
                    internal_option,
                    source_val,
                    rec_duration,
                    actual_model_type,
                    use_vision=False
                )
        else:
            with st.spinner(f"{T['run_btn']}..."):
                # Pass the 'use_vision_enabled' state to the pipeline
                res_id = run_pipeline(
                    internal_option,
                    source_val,
                    30,
                    actual_model_type,
                    use_vision=use_vision_enabled
                )

        # Handle successful services
        if res_id:
            st.success("OK!")
            st.rerun()

# --- MAIN UI ---
st.title(T["title"])

# Prepare source mapping for the multiselect component
source_mapping = get_human_readable_sources()

# FIX: Added 'placeholder' and translated source selection label
selected_names = st.multiselect(
    T.get("source_select", "Sources:"),
    options=list(source_mapping.keys()),
    placeholder=T.get("choose_source", "Choose options")
)
selected_ids = [source_mapping[name] for name in selected_names]

if selected_ids:
    chat_col, trans_col = st.columns([1, 1])
    insight_engine = get_insight_engine()
    lang_full = {"PL": "Polish", "RU": "Russian", "EN": "English"}[st.session_state.ui_lang]

    with chat_col:
        st.subheader(T["chat_header"])
        chat_box = st.container(height=350)
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        # Interactive AI chat logic
        if prompt := st.chat_input(T["chat_placeholder"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            res = insight_engine.ask(prompt, filter_sources=selected_ids, target_lang=lang_full)
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            st.rerun()

        st.divider()
        # Cleaned management layout (removed translate button from here)
        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            if st.button(T["mom_btn"], use_container_width=True):
                raw_mom = insight_engine.generate_mom(selected_ids[0], target_lang=lang_full)
                st.session_state.last_analysis = prepare_content_for_editor(raw_mom)
                st.rerun()

        with c2:
            if st.button(T["audit_btn"], use_container_width=True):
                raw_audit = insight_engine.analyze_audit_details(selected_ids[0], target_lang=lang_full)
                st.session_state.last_analysis = prepare_content_for_editor(raw_audit)
                st.rerun()

        with c3:
            if st.button(T["del_btn"], use_container_width=True, type="secondary"):
                delete_source_data(selected_ids[0], selected_names[0])
                st.rerun()

        # Draft editor for audit reporting
        if st.session_state.last_analysis:
            st.subheader("🖋️ Draft Editor")
            edited_report = st_quill(
                value=st.session_state.last_analysis,
                html=True,
                toolbar=[['bold', 'italic'], [{'list': 'ordered'}, {'list': 'bullet'}], ['clean']]
            )

            e1, e2 = st.columns(2)
            with e1:
                if st.button(T["save_btn"], use_container_width=True):
                    save_draft(selected_names[0], edited_report)
                    st.success("Saved!")
            with e2:
                if st.button(T["export_btn"], use_container_width=True):
                    rep = AuditReporter(FONT_DIR, DATA_DIRS["reports"])
                    # Defaulting user name to Natalia as per background info
                    path = rep.generate_pdf_report(selected_names[0], edited_report, "Natalia")
                    with open(path, "rb") as f:
                        st.download_button("📥 Download PDF", f, file_name=path.name)

    with trans_col:
        st.subheader(T["trans_header"])
        v_id = selected_ids[0]
        p_file = DATA_DIRS["processed"] / f"{v_id}_processed.json"
        frame_dir = DATA_DIRS["frames"] / v_id

        if p_file.exists():
            with open(p_file, "r", encoding="utf-8") as f:
                processed_data = json.load(f)

            # --- TRANSLATION LOGIC ---
            current_lang = st.session_state.ui_lang
            translations = processed_data.get("translations", {})
            has_translation = current_lang in translations

            c_head, c_btn = st.columns([1, 1])
            if has_translation:
                show_original = c_head.toggle("Show original text", value=False)
            else:
                if c_btn.button(f"🌐 Translate to {current_lang}", use_container_width=True):
                    with st.spinner(f"Translating segments..."):
                        engine = get_insight_engine()
                        lang_name = {"PL": "Polish", "RU": "Russian", "EN": "English"}[current_lang]

                        translated_list = []
                        for seg in processed_data["segments"]:
                            ts = seg.copy()
                            ts['text'] = engine.translate_text(seg['text'], lang_name)
                            translated_list.append(ts)

                        processed_data.setdefault("translations", {})[current_lang] = translated_list
                        with open(p_file, "w", encoding="utf-8") as f:
                            json.dump(processed_data, f, ensure_ascii=False, indent=2)
                        st.rerun()
                show_original = True

            # --- DATA EXPORT ---
            # Using the helper function to generate ZIP (Excel + Frames)
            zip_payload = create_data_package(v_id, processed_data, frame_dir)
            st.download_button(
                label="📥 Download Data Package (Excel + Frames)",
                data=zip_payload,
                file_name=f"Data_Export_{v_id}.zip",
                mime="application/zip",
                use_container_width=True
            )
            st.divider()

            # --- DATA SELECTION ---
            if has_translation and not show_original:
                segments = processed_data["translations"][current_lang]
                display_lang = {"PL": "Polish", "RU": "Russian", "EN": "English"}.get(current_lang)
            else:
                segments = processed_data["segments"]
                display_lang = processed_data.get("language", "Original")

            # --- RENDERING ENGINE (LINEAR LAYOUT) ---
            linker = VisualLinker(frame_dir, get_insight_engine())
            has_frames = any(frame_dir.glob("*.jpg")) if frame_dir.exists() else False
            vision_enabled = processed_data.get("use_vision_flag", True)

            if has_frames and vision_enabled:
                with st.spinner(f"Linking visual assets ({display_lang})..."):
                    anchored_data = linker.get_anchored_frames(segments, detected_lang=display_lang)
            else:
                anchored_data = [{
                    "start": s['start'], "audio_track": s['text'],
                    "frame": None, "video_context": None
                } for s in segments]

            # Linear Layout: Time | Text | Asset
            for item in anchored_data:
                col_t, col_txt, col_media = st.columns([0.12, 0.73, 0.15])

                with col_t:
                    st.caption(f"**{format_timestamp(item['start'])}**")

                with col_txt:
                    st.markdown(item['audio_track'])
                    if item.get('video_context'):
                        st.caption(f"Visual Context: {item['video_context']}")

                with col_media:
                    if item.get('frame'):
                        st.image(item['frame'], use_container_width=True)
                        if st.button("📎", key=f"add_{item['start']}", help="Add to draft"):
                            snippet = f"<p><b>[{format_timestamp(item['start'])}]</b> {item['audio_track']}</p>"
                            st.session_state.last_analysis += snippet
                            st.toast("Snippet added")

                st.write("---")

else:
    # Display localized waiting message
    st.info(T["wait_msg"])