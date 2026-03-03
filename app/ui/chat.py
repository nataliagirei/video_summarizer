import streamlit as st


class ChatUI:
    """
    Handles the interactive AI Chat interface for the Lumina Audit System.
    Optimized for macOS to prevent UI freezing and redundant inputs.
    """

    def __init__(self, insight_engine):
        self.insight_engine = insight_engine

    def render(self, selected_ids: list, target_lang: str = "English"):
        """
        Renders the chat window with immediate feedback and stable message flow.
        """
        # Ensure persistent chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("💬 Chat")

        # 1. CREATE A SCROLLABLE CONTAINER
        # We use a fixed-height container to keep the UI clean
        chat_container = st.container(height=450)

        # 2. DISPLAY EXISTING HISTORY
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 3. HANDLE NEW INPUT
        if prompt := st.chat_input("Ask a question about the audit context..."):

            # --- STEP A: IMMEDIATE FEEDBACK ---
            # We display the user's message inside the container BEFORE the heavy AI logic starts
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Save to session state immediately
            st.session_state.messages.append({"role": "user", "content": prompt})

            # --- STEP B: AI PROCESSING ---
            answer = ""
            try:
                if self.insight_engine and selected_ids:
                    # Show the spinner inside the assistant's message area
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                # RAG query execution
                                response = self.insight_engine.ask(
                                    prompt,
                                    filter_sources=selected_ids,
                                    target_lang=target_lang
                                )

                                # Process and normalize the response
                                if isinstance(response, dict):
                                    answer = response.get(
                                        "answer",
                                        "No relevant information found in the selected transcripts."
                                    )
                                else:
                                    answer = str(response)

                                # Display the answer immediately
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

            # --- STEP C: FINALIZE STATE ---
            # Append assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # FORCE RERUN: This ensures the input box clears and the state is fully synced
            # Without this, Streamlit might feel "stuck" for a moment
            st.rerun()

    def clear_history(self):
        """Resets the audit conversation context."""
        st.session_state.messages = []
        st.rerun()