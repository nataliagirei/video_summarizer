import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from threading import RLock

class DraftEditor:
    """
    A robust Document & Report Editor for Natalia's workflow.
    Supports persistent JSON storage, AI analysis sync, and full localization.
    """

    # RLock prevents file corruption during simultaneous read/write operations
    _file_lock = RLock()

    def __init__(self, reporter, drafts_path: Path):
        """
        Initializes the editor with a PDF reporter and storage path.
        """
        self.reporter = reporter
        self.drafts_file = Path(drafts_path)
        self.drafts_file.parent.mkdir(parents=True, exist_ok=True)

    def load_drafts(self) -> dict:
        """
        Reads all saved drafts from the JSON file with thread safety.
        """
        if not self.drafts_file.exists() or self.drafts_file.stat().st_size == 0:
            return {}

        with self._file_lock:
            try:
                content = self.drafts_file.read_text(encoding="utf-8")
                return json.loads(content) if content else {}
            except Exception as e:
                st.error(f"Draft load error: {e}")
                return {}

    def save_draft(self, title: str, content: str) -> bool:
        """
        Saves the current content to JSON storage.
        """
        if not title:
            return False

        with self._file_lock:
            try:
                drafts = self.load_drafts()
                drafts[title] = {
                    "content": content,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                self.drafts_file.write_text(
                    json.dumps(drafts, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                return True
            except Exception as e:
                st.error(f"Save error: {e}")
                return False

    def delete_draft(self, title: str) -> bool:
        """
        Removes a draft from the database.
        """
        with self._file_lock:
            try:
                drafts = self.load_drafts()
                if title in drafts:
                    del drafts[title]
                    self.drafts_file.write_text(
                        json.dumps(drafts, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    return True
                return False
            except Exception as e:
                st.error(f"Deletion error: {e}")
                return False

    def render_selector(self, T: dict):
        """
        Sidebar component to browse and manage saved documents.
        Uses T for multilingual labels.
        """
        drafts = self.load_drafts()

        if drafts:
            st.subheader(T.get("drafts_header", "📁 Saved Drafts"))

            # Format options for the dropdown: "Title (Date)"
            options = {f"{k} ({v['date']})": k for k, v in drafts.items()}

            selected_label = st.selectbox(
                T.get("select_draft", "Select a draft:"),
                options=list(options.keys()),
                index=None,
                key=f"selector_{len(drafts)}"
            )

            if selected_label:
                real_title = options[selected_label]
                col_load, col_del = st.columns([0.8, 0.2])

                with col_load:
                    if st.button(T.get("load_draft", "📂 Load"), width="stretch"):
                        # Update session state to trigger UI refresh
                        st.session_state.editor_content = drafts[real_title]["content"]
                        st.session_state.current_draft_title = real_title
                        st.session_state.prev_synced_val = drafts[real_title]["content"]
                        st.session_state.editor_trigger = st.session_state.get("editor_trigger", 0) + 1
                        st.rerun()

                with col_del:
                    # Trash icon with tooltip
                    if st.button("🗑️", width="stretch", help=T.get("delete_draft_btn", "Delete")):
                        if self.delete_draft(real_title):
                            st.toast(T.get("save_success", "Removed."))
                            st.rerun()

    def render(self, T: dict, source_name: str = "New_Document"):
        """
        Main editor interface with AI sync and PDF generation.
        """
        # ---------- INITIALIZE SESSION STATES ----------
        if "editor_content" not in st.session_state:
            st.session_state.editor_content = ""
        if "current_draft_title" not in st.session_state:
            st.session_state.current_draft_title = source_name
        if "prev_synced_val" not in st.session_state:
            st.session_state.prev_synced_val = ""
        if "editor_trigger" not in st.session_state:
            st.session_state.editor_trigger = 0

        # ---------- SYNC WITH AI ANALYSIS ----------
        # If the AI produces a new report/summary, it updates the editor automatically
        new_ai_text = st.session_state.get("last_analysis", "")
        if new_ai_text and new_ai_text != st.session_state.prev_synced_val:
            st.session_state.editor_content = new_ai_text
            st.session_state.prev_synced_val = new_ai_text
            st.session_state.editor_trigger += 1
            st.rerun()

        st.markdown(f"### {T.get('editor_header', '📝 Editor')}")

        # ---------- INPUT FIELDS ----------
        # Dynamic keys ensure that loading a draft actually resets the text fields
        st.session_state.current_draft_title = st.text_input(
            T.get("draft_filename", "Filename:"),
            value=st.session_state.current_draft_title,
            key=f"title_in_{st.session_state.editor_trigger}"
        )

        edited_text = st.text_area(
            T.get("content_label", "Content:"),
            value=st.session_state.editor_content,
            height=500,
            key=f"area_{st.session_state.editor_trigger}"
        )

        # Update session state as Natalia types
        if edited_text != st.session_state.editor_content:
            st.session_state.editor_content = edited_text

        # ---------- ACTION BUTTONS ----------
        save_triggered = False
        col1, col2 = st.columns(2)

        with col1:
            if st.button(T.get("save_draft_btn", "💾 SAVE"), width="stretch", type="primary"):
                save_triggered = True

        with col2:
            if st.button(T.get("gen_pdf_btn", "📄 PDF"), width="stretch"):
                with st.spinner("Building PDF..."):
                    # Generate PDF using the reporter service
                    path = self.reporter.generate_pdf_report(
                        st.session_state.current_draft_title,
                        st.session_state.editor_content,
                        "Natalia" # Author name
                    )
                    st.session_state.pdf_path = path
                    st.success(T.get("pdf_success", "Done!"))

        # ---------- EXECUTE SAVE ----------
        if save_triggered:
            if self.save_draft(st.session_state.current_draft_title, st.session_state.editor_content):
                st.session_state.prev_synced_val = st.session_state.editor_content
                st.toast(T.get("save_success", "Saved!"))
                st.rerun()

        # ---------- DOWNLOAD SECTION ----------
        if st.session_state.get("pdf_path"):
            pdf_path = Path(st.session_state.pdf_path)
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        T.get("download_pdf_btn", "⬇️ Download"),
                        f,
                        file_name=f"{st.session_state.current_draft_title}.pdf",
                        width="stretch"
                    )