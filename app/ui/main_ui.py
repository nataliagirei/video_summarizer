from pathlib import Path

import streamlit as st

from app.ui.ui_lang import LANG_DICT
# Internal module imports
from chat import ChatUI
from editor import DraftEditor
from pipeline import Pipeline
from services.processing.export_pdf import PDFReporter

# --- DIRECTORY CONFIGURATION ---
# Ensures all necessary folders exist before the app starts
DATA_DIRS = {k: Path(f"data/{k}") for k in ["raw", "audio", "transcripts", "processed", "reports", "frames"]}
for d in DATA_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("assets/fonts")
DRAFTS_PATH = DATA_DIRS["raw"] / "drafts.json"


# --- CACHING STRATEGY ---
@st.cache_resource
def get_pipeline(model_type):
    """
    Initializes the Pipeline service once and caches it.
    Prevents reloading heavy Whisper models on every UI interaction.
    """
    return Pipeline(DATA_DIRS, whisper_model=model_type)


def run_ui():
    """
    Main function to render the Lumina AI User Interface.
    Orchestrates Sidebar, Chat, Editor, and Transcript views with full localization.
    """
    st.set_page_config(page_title="Lumina AI Assistant", layout="wide")

    # --- SESSION STATE INITIALIZATION ---
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "EN"
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = ""
    if "processing_source" not in st.session_state:
        st.session_state.processing_source = False

    # Get active translation dictionary based on selected language
    T = LANG_DICT[st.session_state.ui_lang]
    # Store T in session state for ChatUI and other components to access easily
    st.session_state.T_dict = T

    # --- UI SERVICE INITIALIZATION ---
    reporter = PDFReporter(FONT_DIR, DATA_DIRS["reports"])
    draft_editor = DraftEditor(reporter, DRAFTS_PATH)

    # --- SIDEBAR: CONFIGURATION & INPUTS ---
    with st.sidebar:
        # 1. Global Language Selector
        st.session_state.ui_lang = st.selectbox(
            "🌐 Interface Language / Język",
            ["PL", "RU", "EN"],
            index=["PL", "RU", "EN"].index(st.session_state.ui_lang)
        )

        # Refresh T after language change
        T = LANG_DICT[st.session_state.ui_lang]
        st.header(T["sidebar_header"])

        # 2. Model Quality Selection
        quality_label = st.radio(
            T["quality_option"],
            [T["quality_fast"], T["quality_high"]]
        )

        quality_map = {
            T["quality_fast"]: "base",
            T["quality_high"]: "medium"
        }

        # Initialize the pipeline core
        pipeline = get_pipeline(quality_map[quality_label])

        # Sync the engine's internal language with the current UI selection
        pipeline.insight_engine.ui_lang = st.session_state.ui_lang

        chat_ui = ChatUI(pipeline.insight_engine)

        # 3. Source Selection (Multiselect)
        source_mapping = pipeline.get_human_readable_sources()
        selected_names = st.multiselect(
            T["source_select"],
            options=list(source_mapping.keys()),
            default=list(source_mapping.keys())[:1] if source_mapping else None
        )
        selected_ids = [source_mapping[name] for name in selected_names] if selected_names else []

        # 4. Source Deletion Logic (Localized)
        if selected_ids:
            if st.button("🗑️ " + T["delete_btn"], width="stretch"):
                if pipeline.delete_source(selected_ids[0]):
                    st.toast(T["save_success"])  # Generic success message
                    st.rerun()

        st.divider()

        # 5. Saved Drafts Management
        draft_editor.render_selector(T)

        st.divider()

        # 6. Data Input Method Selection
        methods_map = {
            "YouTube": "YouTube",
            T["method_file"]: "File",
            T["method_mic"]: "Microphone"
        }

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
            source_value = st.text_input("YouTube URL:")

        # Pipeline Trigger
        if st.button(
                T["run_btn"],
                width="stretch",
                type="primary",
                disabled=st.session_state.processing_source
        ):
            st.session_state.processing_source = True
            st.rerun()

    # --- PIPELINE EXECUTION (ASYNC WORKER) ---
    if st.session_state.processing_source:
        with st.spinner("Lumina AI is analyzing the source..."):
            pipeline.run(
                internal_method,
                source_value,
                duration=rec_duration if internal_method == "Microphone" else 30,
                use_vision=use_vision
            )
        st.session_state.processing_source = False
        st.rerun()

    # --- MAIN CONTENT AREA: DUAL COLUMN LAYOUT ---
    left_col, right_col = st.columns([1.2, 1])

    # LEFT COLUMN: AI Interaction & Drafting
    with left_col:
        # Chat for contextual queries
        chat_ui.render(selected_ids=selected_ids)

        st.divider()

        # --- QUICK ANALYSIS TOOLS (Localized) ---
        if selected_ids:
            c1, c2, c3 = st.columns(3)

            # 1. Summary / Minutes
            with c1:
                if st.button(f"📄 {T['mom_btn']}", width="stretch"):
                    with st.spinner("Processing..."):
                        res = pipeline.insight_engine.generate_summary(
                            selected_ids[0],
                            target_lang=st.session_state.ui_lang
                        )
                        st.session_state.last_analysis = res

            # 2. Export Package (ZIP)
            with c2:
                zip_data = pipeline.prepare_export(selected_ids[0])
                if zip_data:
                    st.download_button(
                        label=f"⬇️ {T['zip_btn']}",
                        data=zip_data,
                        file_name=f"Project_{selected_ids[0]}.zip",
                        mime="application/zip",
                        width="stretch"
                    )

            # 3. Key Insights / Sentiment
            with c3:
                if st.button(f"📊 {T['audit_btn']}", width="stretch"):
                    with st.spinner("Analyzing..."):
                        res = pipeline.insight_engine.analyze_key_insights(
                            selected_ids[0],
                            target_lang=st.session_state.ui_lang
                        )
                        st.session_state.last_analysis = res

        # Professional Document & Report Editor
        current_source_id = selected_ids[0] if selected_ids else "New_Project"
        draft_editor.render(T, source_name=current_source_id)

    # RIGHT COLUMN: Visual Evidence & Interactive Transcript
    with right_col:
        header_row, toggle_row = st.columns([0.6, 0.4])

        with header_row:
            st.subheader(T["trans_header"])

        if selected_ids:
            with toggle_row:
                # Localized toggle for original text (supports RU, PL, EN)
                toggle_label = T.get("toggle_original", "Show Original")
                show_original = st.toggle(toggle_label, value=False, key=f"main_toggle_{selected_ids[0]}")

            # Fixed-height container for the transcript
            with st.container(height=800, border=True):
                pipeline.render_transcript_view(
                    selected_ids[0],
                    T,
                    ui_lang=st.session_state.ui_lang,
                    show_original=show_original
                )
        else:
            st.info(T["wait_msg"])


if __name__ == "__main__":
    run_ui()
