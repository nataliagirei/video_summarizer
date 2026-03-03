import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from threading import Lock


class DraftEditor:
    _file_lock = Lock()

    def __init__(self, reporter, drafts_path: Path):
        self.reporter = reporter
        self.drafts_file = Path(drafts_path)
        self.drafts_file.parent.mkdir(parents=True, exist_ok=True)

    def load_drafts(self) -> dict:
        if not self.drafts_file.exists():
            return {}
        with self._file_lock:
            try:
                # Читаем напрямую через Path для скорости
                content = self.drafts_file.read_text(encoding="utf-8")
                return json.loads(content) if content else {}
            except:
                return {}

    def save_draft(self, title: str, content: str):
        if not title:
            return False
        with self._file_lock:
            try:
                drafts = self.load_drafts()
                drafts[title] = {
                    "content": content,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                self.drafts_file.write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8")
                return True
            except Exception as e:
                st.error(f"Save error: {e}")
                return False

    def render_selector(self, T):
        drafts = self.load_drafts()
        if drafts:
            st.subheader("📁 " + T.get("drafts_header", "Saved Drafts"))
            options = {f"{k} ({v['date']})": k for k, v in drafts.items()}
            # Используем уникальный ключ, зависящий от количества черновиков
            selected_label = st.selectbox(
                "Select a draft:",
                options=list(options.keys()),
                index=None,
                key=f"selector_{len(drafts)}"
            )

            if selected_label:
                real_title = options[selected_label]
                if st.button("📂 Load Draft", width="stretch"):
                    st.session_state.editor_content = drafts[real_title]["content"]
                    st.session_state.current_draft_title = real_title
                    st.session_state.prev_synced_val = drafts[real_title]["content"]
                    st.session_state.editor_trigger = st.session_state.get("editor_trigger", 0) + 1
                    st.rerun()

    def render(self, source_name: str = "New_Report"):
        # 1. State
        if "editor_content" not in st.session_state:
            st.session_state.editor_content = ""
        if "current_draft_title" not in st.session_state:
            st.session_state.current_draft_title = source_name
        if "prev_synced_val" not in st.session_state:
            st.session_state.prev_synced_val = ""
        if "editor_trigger" not in st.session_state:
            st.session_state.editor_trigger = 0

        # 2. AI Sync
        new_ai_text = st.session_state.get("last_analysis", "")
        if new_ai_text and new_ai_text != st.session_state.prev_synced_val:
            st.session_state.editor_content = new_ai_text
            st.session_state.prev_synced_val = new_ai_text
            st.session_state.editor_trigger += 1
            st.rerun()

        st.markdown("### 📝 Audit Report Editor")

        # 3. Title
        st.session_state.current_draft_title = st.text_input(
            "Draft Filename:",
            value=st.session_state.current_draft_title,
            key=f"title_in_{st.session_state.editor_trigger}"
        )

        # 4. Text Area
        edited_text = st.text_area(
            "Content:",
            value=st.session_state.editor_content,
            height=500,
            key=f"area_{st.session_state.editor_trigger}"
        )

        if edited_text != st.session_state.editor_content:
            st.session_state.editor_content = edited_text

        # 5. Buttons Logic
        do_save = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 SAVE DRAFT", width="stretch", type="primary"):
                do_save = True  # Ставим флаг, но не вызываем rerun внутри колонки

        with col2:
            if st.button("📄 GENERATE PDF", width="stretch"):
                with st.spinner("Processing..."):
                    path = self.reporter.generate_pdf_report(
                        st.session_state.current_draft_title,
                        st.session_state.editor_content,
                        "Natalia"
                    )
                    st.session_state.pdf_path = path
                    st.success("PDF Ready!")

        # 6. Execute Save outside of columns to prevent UI freeze
        if do_save:
            if self.save_draft(st.session_state.current_draft_title, st.session_state.editor_content):
                st.session_state.prev_synced_val = st.session_state.editor_content
                st.toast("✅ Saved!")
                st.rerun()  # Теперь это безопасно

        if st.session_state.get("pdf_path"):
            pdf_file = Path(st.session_state.pdf_path)
            if pdf_file.exists():
                with open(pdf_file, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name=f"{st.session_state.current_draft_title}.pdf",
                                       width="stretch")