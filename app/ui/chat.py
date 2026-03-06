import streamlit as st


def detect_lang(text: str) -> str:
    """
    Simple heuristic to detect language based on the first character.
    Returns one of: "Russian", "Polish", "English".
    """
    if not text:
        return "English"
    first_char = text[0].lower()
    if first_char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
        return "Russian"
    elif first_char in "ąćęłńóśżź":
        return "Polish"
    else:
        return "English"


class ChatUI:
    """
    Handles the interactive AI Chat interface for the Lumina Audit System.
    Optimized for macOS to prevent UI freezing and redundant inputs.
    Now supports automatic chat language detection based on user input.
    """

    def __init__(self, insight_engine):
        self.insight_engine = insight_engine

    def render(self, selected_ids: list, default_lang: str = "English"):
        """
        Renders the chat window with immediate feedback and stable message flow.

        Parameters:
        - selected_ids: list of source IDs to query
        - default_lang: fallback language if detection fails
        """
        # Ensure persistent chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("💬 Chat")

        # 1. CREATE A SCROLLABLE CONTAINER
        chat_container = st.container(height=450)

        # 2. DISPLAY EXISTING HISTORY
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 3. HANDLE NEW INPUT
        if prompt := st.chat_input("Ask a question about the audit context..."):

            # --- STEP A: IMMEDIATE FEEDBACK ---
            # Show user message before AI processing
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Save user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})

            # --- STEP B: DETECT LANGUAGE ---
            chat_lang = detect_lang(prompt) or default_lang

            # --- STEP C: AI PROCESSING ---
            answer = ""
            try:
                if self.insight_engine and selected_ids:
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                # Ask AI engine using detected language
                                response = self.insight_engine.ask(
                                    prompt,
                                    filter_sources=selected_ids,
                                    target_lang=chat_lang
                                )

                                # Normalize the response to string
                                if isinstance(response, dict):
                                    answer = response.get(
                                        "answer",
                                        "No relevant information found in the selected transcripts."
                                    )
                                else:
                                    answer = str(response)

                                # Display assistant response immediately
                                st.markdown(answer)

                elif not selected_ids:
                    answer = "⚠️ Please select at least one source in the sidebar first."
                    st.warning(answer)
                else:
                    answer = "AI Engine is not properly initialized."
                    st.error(answer)

            except Exception as e:
                answer = f"Chat backend error: {str(e)}"
                st.error(answer)

            # --- STEP D: FINALIZE STATE ---
            # Append assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Force Streamlit rerun to clear input and sync state
            st.rerun()

    def clear_history(self):
        """Resets the audit conversation context."""
        st.session_state.messages = []
        st.rerun()