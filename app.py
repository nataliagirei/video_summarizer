import streamlit as st
import os
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
from streamlit_quill import st_quill

# --- INITIAL CONFIG & SECURITY ---
load_dotenv()

# --- MODULE IMPORTS ---
from ingestion.youtube_downloader import YouTubeDownloader
from ingestion.audio_extractor import AudioExtractor
from ingestion.microphone_recorder import AudioRecorder
from asr.whisper_transcriber import WhisperTranscriber
from processing.transcript_processor import TranscriptProcessor
from processing.text_chunk import TextChunker
from processing.vector_store import LocalVectorStore
from processing.ask_it import VideoInsight

# --- DIRECTORIES CONFIGURATION ---
DATA_DIRS = {k: Path(f"data/{k}") for k in ["raw", "audio", "transcripts", "processed", "reports"]}
FONT_DIR = Path("assets/fonts")
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
    # We keep markdown-like stars if they exist, or clean them if needed
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
            except json.JSONDecodeError:
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
        self.font_dir = font_dir
        self.output_dir = output_dir

    def generate_pdf_report(self, source_title, content, user="Natalia"):
        # 1. Setup PDF and Fonts
        self.add_page()

        dejavu_reg = self.font_dir / "DejaVuSans.ttf"
        dejavu_bold = self.font_dir / "DejaVuSans-Bold.ttf"

        if dejavu_reg.exists():
            self.add_font("DejaVu", "", str(dejavu_reg), uni=True)
            self.add_font("DejaVu", "B", str(dejavu_bold), uni=True)
            font_main = "DejaVu"
        else:
            font_main = "Arial"

        # 2. Header Section
        self.set_font(font_main, 'B', 16)
        # multi_cell prevents long titles from being cut off
        self.multi_cell(0, 10, f"Audit Report: {source_title}", align='C')

        self.set_font(font_main, '', 10)
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Specialist: {user}", ln=True)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(5)

        # 3. Content Section
        self.set_font(font_main, '', 11)
        self.write_formatted_html(content, font_main)

        # 4. Save file
        report_name = f"Lumina_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = self.output_dir / report_name
        self.output(str(report_path))
        return report_path

    def write_formatted_html(self, html, font_name):
        """Processes HTML tags from Quill and writes to PDF without clipping."""
        # Pre-processing tags
        html = html.replace('<p>', '').replace('</p>', '\n')
        html = html.replace('<ul>', '').replace('</ul>', '')
        html = html.replace('<li>', '\n • ').replace('</li>', '')
        html = html.replace('<br>', '\n').replace('<br/>', '\n')

        parts = re.split(r'(<[^>]+>)', html)

        for part in parts:
            if not part: continue

            if part in ['<b>', '<strong>']:
                self.set_font(font_name, 'B')
            elif part in ['</b>', '</strong>']:
                self.set_font(font_name, '')
            elif part.startswith('<'):
                # Skip unsupported tags (italic/underline) to prevent crashes
                continue
            else:
                # Security: check for page break before writing long chunks
                if self.get_y() > 270:
                    self.add_page()
                self.write(7, part)


# --- UI TRANSLATION DICTIONARY ---
LANG_DICT = {
    "EN": {
        "title": "Lumina", "sidebar_header": "⚙️ Management", "option": "Method:",
        "run_btn": "🚀 Process", "chat_header": "💬 Chat", "chat_placeholder": "Ask...",
        "trans_header": "📄 Transcript", "wait_msg": "👈 Add source.", "quality_option": "Model:",
        "mom_btn": "📝 Minutes", "audit_btn": "🔍 Audit", "export_btn": "📥 Export PDF",
        "del_btn": "🗑️ Delete", "drafts_header": "📝 Drafts", "save_btn": "💾 Save Draft",
        "open_btn": "📂 Open", "source_select": "📚 Sources:"
    },
    "RU": {
        "title": "Lumina", "sidebar_header": "⚙️ Управление", "option": "Метод:",
        "run_btn": "🚀 Обработать", "chat_header": "💬 Чат", "chat_placeholder": "Спросите...",
        "trans_header": "📄 Транскрипт", "wait_msg": "👈 Добавьте источник.", "quality_option": "Модель:",
        "mom_btn": "📝 Протокол", "audit_btn": "🔍 Аудит", "export_btn": "📥 Экспорт PDF",
        "del_btn": "🗑️ Удалить", "drafts_header": "📝 Черновики", "save_btn": "💾 Сохранить",
        "open_btn": "📂 Открыть", "source_select": "📚 Источники:"
    },
    "PL": {
        "title": "Lumina", "sidebar_header": "⚙️ Zarządzanie", "option": "Metoda:",
        "run_btn": "🚀 Przetwórz", "chat_header": "💬 Czat", "chat_placeholder": "Zapytaj...",
        "trans_header": "📄 Transkrypcja", "wait_msg": "👈 Dodaj źródło.", "quality_option": "Model:",
        "mom_btn": "📝 Protokół", "audit_btn": "🔍 Audyt", "export_btn": "📥 Eksport PDF",
        "del_btn": "🗑️ Usuń źródło", "drafts_header": "📝 Szkice", "save_btn": "💾 Zapisz szkic",
        "open_btn": "📂 Otwórz", "source_select": "📚 Źródła:"
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


# --- PIPELINE ---
def run_pipeline(source_type, source_value, duration, model_type):
    try:
        if source_type == "YouTube":
            metadata = YouTubeDownloader(DATA_DIRS["raw"]).download(source_value)
            source_id = metadata["video_id"]
            audio_path = AudioExtractor(DATA_DIRS["audio"]).extract(Path(metadata["filepath"]))
        elif source_type == "Microphone":
            audio_path = AudioRecorder(DATA_DIRS["audio"]).record(duration=duration)
            source_id = audio_path.stem
            registry = {}
            if REGISTRY_FILE.exists():
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f: registry = json.load(f)
            registry[source_id] = {"title": f"Rec {datetime.now().strftime('%H:%M')}", "type": "audio",
                                   "date": datetime.now().isoformat()}
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
        else:
            source_id = source_value.name.split('.')[0]
            temp_path = DATA_DIRS["raw"] / source_value.name
            with open(temp_path, "wb") as f:
                f.write(source_value.getbuffer())
            audio_path = AudioExtractor(DATA_DIRS["audio"]).extract(temp_path) if source_value.type.startswith(
                "video") else temp_path

        whisper_engine = get_whisper_model(model_type)
        transcript = whisper_engine.transcribe(audio_path)
        if not transcript["segments"]: return None

        TranscriptProcessor(DATA_DIRS["processed"]).process(transcript, source_id)
        chunker = TextChunker(max_tokens=150, overlap_tokens=30)
        chunks = chunker.chunk_by_segments(transcript["segments"], transcript.get("language", "en"))

        prepared_chunks = [{"id": f"{source_id}_{i}", "text": ch.text, "vector_text": ch.vector_text,
                            "metadata": {"source": source_id, "start": ch.start, "end": ch.end}} for i, ch in
                           enumerate(chunks)]

        store = get_vector_store()
        store.add_chunks(prepared_chunks)
        store.persist()
        return source_id
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# --- UI APP CONFIG ---
st.set_page_config(page_title="Lumina", layout="wide")
if "ui_lang" not in st.session_state: st.session_state.ui_lang = "PL"
if "last_analysis" not in st.session_state: st.session_state.last_analysis = ""
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.session_state.ui_lang = st.selectbox("🌐 Language", ["PL", "RU", "EN"],
                                            index=["PL", "RU", "EN"].index(st.session_state.ui_lang))
    T = LANG_DICT[st.session_state.ui_lang]
    st.header(T["sidebar_header"])

    st.subheader(T["drafts_header"])
    all_drafts = load_drafts()
    if all_drafts:
        sel_draft = st.selectbox("Load Draft:", options=list(all_drafts.keys()))
        if st.button(T["open_btn"]):
            st.session_state.last_analysis = all_drafts[sel_draft]["content"]

    st.divider()
    internal_option = st.radio(T["option"], ["YouTube", "File", "Microphone"])
    source_val = st.text_input("Link:") if internal_option == "YouTube" else st.file_uploader(
        "File:") if internal_option == "File" else None
    quality_mode = st.radio(T["quality_option"], ("Fast (base)", "High (medium)"))

    if st.button(T["run_btn"], use_container_width=True):
        if run_pipeline(internal_option, source_val, 30, "base" if "Fast" in quality_mode else "medium"):
            st.rerun()

# --- MAIN UI ---
st.title(T["title"])
source_mapping = get_human_readable_sources()
selected_names = st.multiselect(T.get("source_select", "Sources:"), options=list(source_mapping.keys()))
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
                with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input(T["chat_placeholder"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            res = insight_engine.ask(prompt, filter_sources=selected_ids, target_lang=lang_full)
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            st.rerun()

        st.divider()
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
                    path = rep.generate_pdf_report(selected_names[0], edited_report, "Natalia")
                    with open(path, "rb") as f:
                        st.download_button("📥 Download PDF", f, file_name=path.name)

    with trans_col:
        st.subheader(T["trans_header"])
        v_id = source_mapping[selected_names[0]]
        p_file = DATA_DIRS["processed"] / f"{v_id}_processed.json"
        if p_file.exists():
            with open(p_file, "r", encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f)["segments"])
                df['time'] = df['start'].apply(format_timestamp)
                st.dataframe(df[['time', 'text']], hide_index=True, use_container_width=True, height=600)
else:
    st.info(T["wait_msg"])