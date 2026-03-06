import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from threading import RLock  # Reentrant lock to prevent deadlocks during nested file operations


class DraftEditor:
    """
    A robust Audit Report Editor that handles report drafting,
    auto-syncing with AI analysis, and persistent JSON storage.
    """

    # RLock allows the same thread to acquire the lock multiple times,
    # which is crucial if save_draft calls load_drafts internally.
    _file_lock = RLock()

    def __init__(self, reporter, drafts_path: Path):
        self.reporter = reporter
        self.drafts_file = Path(drafts_path)

        # Ensure the directory structure exists on initialization
        self.drafts_file.parent.mkdir(parents=True, exist_ok=True)

    def load_drafts(self) -> dict:
        """
        Retrieves all saved drafts from the JSON file.
        Returns an empty dictionary if the file is missing, empty, or corrupted.
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
        Persists a draft into the JSON storage.
        Prevents race conditions using a thread-safe lock.
        """
        if not title:
            return False

        with self._file_lock:
            try:
                drafts = self.load_drafts()
                # Store content with a timestamp for version tracking
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
        Removes a specific draft from the JSON file by its title.
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
                st.error(f"Error during draft deletion: {e}")
                return False

    def render_selector(self, T: dict):
        """
        Renders the dropdown menu and action buttons (Load/Delete)
        in the Streamlit sidebar or main area.
        """
        drafts = self.load_drafts()

        if drafts:
            st.subheader("📁 " + T.get("drafts_header", "Saved Drafts"))

            # Create a display mapping: "Title (Timestamp)" -> "Title"
            options = {f"{k} ({v['date']})": k for k, v in drafts.items()}

            # Use a dynamic key based on draft count to force UI refresh after deletion
            selected_label = st.selectbox(
                "Select a draft:",
                options=list(options.keys()),
                index=None,
                key=f"selector_{len(drafts)}"
            )

            if selected_label:
                real_title = options[selected_label]

                # Layout for Load and Delete buttons
                col_load, col_del = st.columns([0.8, 0.2])

                with col_load:
                    if st.button("📂 Load Draft", width="stretch"):
                        st.session_state.editor_content = drafts[real_title]["content"]
                        st.session_state.current_draft_title = real_title
                        st.session_state.prev_synced_val = drafts[real_title]["content"]
                        # Increment trigger to force text_area to update with new value
                        st.session_state.editor_trigger = st.session_state.get("editor_trigger", 0) + 1
                        st.rerun()

                with col_del:
                    if st.button("🗑️", width="stretch", help="Delete this draft permanently"):
                        if self.delete_draft(real_title):
                            st.toast(f"Draft '{real_title}' removed successfully.")
                            st.rerun()

    def render(self, source_name: str = "New_Report"):
        """
        The main Editor UI. Includes title input, text area, and action buttons.
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
        # Automatically update the editor if new AI analysis results arrive
        new_ai_text = st.session_state.get("last_analysis", "")
        if new_ai_text and new_ai_text != st.session_state.prev_synced_val:
            st.session_state.editor_content = new_ai_text
            st.session_state.prev_synced_val = new_ai_text
            st.session_state.editor_trigger += 1
            st.rerun()

        st.markdown("### 📝 Audit Report Editor")

        # ---------- INPUT FIELDS ----------
        # Key uses editor_trigger to force widget reset when a draft is loaded
        st.session_state.current_draft_title = st.text_input(
            "Draft Filename:",
            value=st.session_state.current_draft_title,
            key=f"title_in_{st.session_state.editor_trigger}"
        )

        edited_text = st.text_area(
            "Content:",
            value=st.session_state.editor_content,
            height=500,
            key=f"area_{st.session_state.editor_trigger}"
        )

        # Update local session state on user typing
        if edited_text != st.session_state.editor_content:
            st.session_state.editor_content = edited_text

        # ---------- ACTION BUTTONS ----------
        save_triggered = False
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 SAVE DRAFT", width="stretch", type="primary"):
                save_triggered = True

        with col2:
            if st.button("📄 GENERATE PDF", width="stretch"):
                with st.spinner("Building PDF report..."):
                    path = self.reporter.generate_pdf_report(
                        st.session_state.current_draft_title,
                        st.session_state.editor_content,
                        "Natalia"
                    )
                    st.session_state.pdf_path = path
                    st.success("PDF generated successfully!")

        # ---------- EXECUTE SAVE ----------
        # Logic is outside columns to prevent Streamlit layout glitches
        if save_triggered:
            if self.save_draft(st.session_state.current_draft_title, st.session_state.editor_content):
                st.session_state.prev_synced_val = st.session_state.editor_content
                st.toast("✅ Draft saved to database.")
                st.rerun()

        # ---------- DOWNLOAD SECTION ----------
        if st.session_state.get("pdf_path"):
            pdf_path = Path(st.session_state.pdf_path)
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        f,
                        file_name=f"{st.session_state.current_draft_title}.pdf",
                        width="stretch"
                    )