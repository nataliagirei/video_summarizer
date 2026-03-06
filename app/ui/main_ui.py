import streamlit as st
from pathlib import Path
import json

# Internal module imports
from chat import ChatUI, detect_lang
from editor import DraftEditor
from services.processing.export_pdf import PDFReporter
from pipeline import Pipeline
from app.ui.ui_lang import LANG_DICT


# --- DIRECTORY CONFIGURATION ---
DATA_DIRS = {k: Path(f"data/{k}") for k in ["raw", "audio", "transcripts", "processed", "reports", "frames"]}
for d in DATA_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("assets/fonts")
DRAFTS_PATH = DATA_DIRS["raw"] / "drafts.json"

# --- LANGUAGE MAP FOR LLM ---
LANG_MAP = {"PL": "Polish", "RU": "Russian", "EN": "English"}

# --- CACHING STRATEGY ---
@st.cache_resource
def get_pipeline(model_type):
    return Pipeline(DATA_DIRS, whisper_model=model_type)


def run_ui():
    st.set_page_config(page_title="Lumina Audit AI", layout="wide")

    # --- SESSION STATE INITIALIZATION ---
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "PL"
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = ""
    if "processing_source" not in st.session_state:
        st.session_state.processing_source = False

    # --- UI SERVICE INITIALIZATION ---
    reporter = PDFReporter(FONT_DIR, DATA_DIRS["reports"])
    draft_editor = DraftEditor(reporter, DRAFTS_PATH)

    # --- SIDEBAR: CONTROLS & SETTINGS ---
    with st.sidebar:
        st.session_state.ui_lang = st.selectbox(
            "🌐 Language / Język",
            ["PL", "RU", "EN"],
            index=["PL", "RU", "EN"].index(st.session_state.ui_lang)
        )
        T = LANG_DICT[st.session_state.ui_lang]

        st.header(T["sidebar_header"])

        # Model quality selection
        quality_label = st.radio(
            T["quality_option"],
            [T["quality_fast"], T["quality_high"]]
        )
        quality_map = {T["quality_fast"]: "base", T["quality_high"]: "medium"}

        pipeline = get_pipeline(quality_map[quality_label])
        chat_ui = ChatUI(pipeline.insight_engine)

        # --- SOURCE MANAGEMENT ---
        source_mapping = pipeline.get_human_readable_sources()
        selected_names = st.multiselect(
            T["source_select"],
            options=list(source_mapping.keys()),
            default=list(source_mapping.keys())[:1] if source_mapping else None
        )
        selected_ids = [source_mapping[name] for name in selected_names]

        if selected_ids:
            if st.button("🗑️ Delete Selected Source", width="stretch", help="Deletes the first selected source and all associated files."):
                target_to_del = selected_ids[0]
                if pipeline.delete_source(target_to_del):
                    st.toast(f"Source {target_to_del} deleted successfully.")
                    st.rerun()

        st.divider()

        # Saved drafts
        draft_editor.render_selector(T)
        st.divider()

        # --- DATA ACQUISITION METHOD ---
        methods_map = {"YouTube": "YouTube", T["method_file"]: "File", T["method_mic"]: "Microphone"}
        selected_method_label = st.radio(T["option"], list(methods_map.keys()))
        internal_method = methods_map[selected_method_label]

        use_vision = st.toggle(T["visual_toggle"], value=True)
        source_value = None
        rec_duration = 30

        if internal_method == "File":
            source_value = st.file_uploader(T["file_label"])
        elif internal_method == "Microphone":
            st.warning(T["mic_warning"])
            rec_duration = st.slider(T["rec_duration"], 5, 120, 30)
            source_value = "MIC_ACTIVE"
        elif internal_method == "YouTube":
            source_value = st.text_input("YouTube Link:")

        if st.button(T["run_btn"], width="stretch", type="primary", disabled=st.session_state.processing_source):
            st.session_state.processing_source = True
            st.rerun()

    # --- PIPELINE PROCESSING LOGIC ---
    if st.session_state.processing_source:
        with st.spinner("Lumina is analyzing the source..."):
            pipeline.run(
                internal_method,
                source_value,
                duration=rec_duration if internal_method == "Microphone" else 30,
                use_vision=use_vision
            )
        st.session_state.processing_source = False
        st.rerun()

    # --- MAIN UI LAYOUT ---
    left_col, right_col = st.columns([1.2, 1])

    # LEFT COLUMN: INTERACTIVE TOOLS
    with left_col:
        chat_ui.render(selected_ids=selected_ids, default_lang=LANG_MAP[st.session_state.ui_lang])
        st.divider()

        if selected_ids:
            c1, c2, c3 = st.columns(3)

            # --- GENERATE MINUTES OF MEETING ---
            with c1:
                if st.button(f"📄 {T['mom_btn']}", width="stretch"):
                    with st.spinner("Generating MoM..."):
                        res = pipeline.insight_engine.generate_mom(selected_ids[0])
                        st.session_state.last_analysis = res

            # --- EXPORT DATA PACKAGE ---
            with c2:
                zip_data = pipeline.prepare_export(selected_ids[0])
                if zip_data:
                    st.download_button(
                        label="Download Audit ZIP",
                        data=zip_data,
                        file_name=f"Audit_{selected_ids[0]}.zip",
                        mime="application/zip",
                        width="stretch"
                    )

            # --- DETAILED AUDIT ANALYSIS ---
            with c3:
                if st.button(f"📊 {T['audit_btn']}", width="stretch"):
                    with st.spinner("Analyzing Audit..."):
                        res = pipeline.insight_engine.analyze_audit_details(selected_ids[0])
                        st.session_state.last_analysis = res

        # Report Editor Section
        current_source_id = selected_ids[0] if selected_ids else "New_Report"
        draft_editor.render(source_name=current_source_id)

    # RIGHT COLUMN: VISUAL TRANSCRIPT
    with right_col:
        st.subheader(T["trans_header"])
        if selected_ids:
            pipeline.render_transcript_view(selected_ids[0], T)
        else:
            st.info(T["wait_msg"])


if __name__ == "__main__":
    run_ui()