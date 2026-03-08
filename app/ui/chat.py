import streamlit as st


class ChatUI:
    """
    General-purpose AI Chat interface.
    Language is fully controlled via st.session_state.ui_lang.
    """

    def __init__(self, insight_engine):
        self.insight_engine = insight_engine

    def render(self, selected_ids: list):
        # Access the translation dictionary based on global state
        T = st.session_state.get("T_dict", {})

        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader(T.get("chat_header", "💬 Chat"))

        chat_container = st.container(height=500)

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Multilingual placeholder for chat input
        if prompt := st.chat_input(T.get("chat_placeholder", "Type here...")):
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})
            current_lang_code = st.session_state.get("ui_lang", "PL")

            answer = ""
            try:
                if self.insight_engine and selected_ids:
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = self.insight_engine.ask(
                                    prompt,
                                    filter_sources=selected_ids,
                                    target_lang=current_lang_code
                                )
                                if isinstance(response, dict):
                                    answer = response.get("answer", "No relevant info.")
                                else:
                                    answer = str(response)
                                st.markdown(answer)

                elif not selected_ids:
                    answer = T.get("chat_error_no_source", "Select a source first.")
                    st.warning(answer)

            except Exception as e:
                answer = f"Error: {str(e)}"
                st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
