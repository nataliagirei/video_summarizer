import streamlit as st
import re


class ChatUI:
    """
    Handles the interactive AI Chat interface for the Lumina Audit System.
    Synchronized with global UI language state to ensure consistent audit reporting.
    """

    def __init__(self, insight_engine):
        """
        Initializes the Chat UI with the RAG engine.
        """
        self.insight_engine = insight_engine

    def render(self, selected_ids: list):
        """
        Renders the chat window and manages message flow using Streamlit session state.
        Now strictly follows st.session_state.ui_lang for AI responses.
        """
        # Ensure persistent chat history exists in the session
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("💬 Audit AI Assistant")

        # 1. CHAT HISTORY CONTAINER
        # Using a fixed height container for better UX during long audit discussions
        chat_container = st.container(height=500)

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 2. CHAT INPUT HANDLING
        # The prompt is captured and processed immediately
        if prompt := st.chat_input("Ask about evidence, risks, or meeting details..."):

            # Display user message in the UI immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Record user message in history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 3. AI LOGIC & LANGUAGE SYNCHRONIZATION
            # We bypass manual detection and use the language Natalya selected in the sidebar
            current_lang_code = st.session_state.get("ui_lang", "PL")

            answer = ""
            try:
                if self.insight_engine and selected_ids:
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing transcripts..."):
                                # Call the RAG engine
                                # We pass target_lang to ensure VideoInsight knows which LANG_MAP to use
                                response = self.insight_engine.ask(
                                    prompt,
                                    filter_sources=selected_ids,
                                    target_lang=current_lang_code
                                )

                                # Handle both dictionary and string return types safely
                                if isinstance(response, dict):
                                    answer = response.get("answer", "No relevant data found.")
                                else:
                                    answer = str(response)

                                st.markdown(answer)

                elif not selected_ids:
                    answer = "⚠️ Audit context missing: Please select a source in the sidebar."
                    st.warning(answer)
                else:
                    answer = "Technical Error: AI Engine not initialized."
                    st.error(answer)

            except Exception as e:
                # Capture backend errors (e.g., API timeouts or Vector Store issues)
                answer = f"Chat backend error: {str(e)}"
                st.error(answer)

            # 4. FINALIZING STATE
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Rerun to clear the input widget and synchronize the visual state
            st.rerun()

    def clear_history(self):
        """
        Clears the conversation context. Useful when switching audit projects.
        """
        st.session_state.messages = []
        st.rerun()