import os
import json
from pathlib import Path
from groq import Groq
from processing.vector_store import LocalVectorStore


class VideoInsight:
    def __init__(self):
        """
        Initialize the insight engine with centralized registry access
        and strict system prompts.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found.")

        self.store = LocalVectorStore()
        self.client = Groq(api_key=self.api_key)
        self.registry_path = Path("data/raw/registry.json")

    def _get_video_title(self, source_id: str) -> str:
        """Helper to map Video ID to Human Readable Title from registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
                return registry.get(source_id, {}).get("title", source_id)
        return source_id

    def _format_time(self, seconds: float) -> str:
        """Converts raw seconds to MM:SS format."""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def analyze_audit_details(self, source_id: str, target_lang: str = "English", model_name="llama-3.3-70b-versatile"):
        """
        Extracts Sentiment and details.
        Fixed: Now respects the requested output language.
        """
        try:
            json_path = Path(f"data/processed/{source_id}_processed.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            context = " ".join([s['text'] for s in data['segments']])[:18000]

            prompt = f"""Analyze this transcript. Provide the response ONLY in {target_lang}.

            1. Sentiment Analysis (Tone of discussion).
            2. Key Action Items (Who/What/When).
            3. Specific mentions of Dates, Values, or Names.

            TRANSCRIPT:
            {context}"""

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are a helpful analyst. Respond in {target_lang}."},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Analysis Error: {str(e)}"

    def generate_mom(self, source_id: str, target_lang: str = "English", model_name="llama-3.3-70b-versatile"):
        """
        Generates Minutes of Meeting.
        Fixed: Now respects the requested output language.
        """
        try:
            json_path = Path(f"data/processed/{source_id}_processed.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            context = " ".join([s['text'] for s in data['segments']])[:18000]

            prompt = f"""Generate a professional Minutes of Meeting (MoM) in {target_lang}.
            Structure: Summary, Discussion Points, Decisions, and Action Plan.

            TRANSCRIPT:
            {context}"""

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": f"You are a professional assistant. You output ONLY in {target_lang}."},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"MoM Error: {str(e)}"

    def ask(self, question: str, filter_sources: list = None, target_lang: str = "English",
            model_name="llama-3.3-70b-versatile"):
        """
        RAG search with Clean Citations and Prompt Injection Protection.
        """
        search_results = self.store.search(question, k=15)
        if filter_sources:
            search_results = [res for res in search_results if res["content"]["metadata"]["source"] in filter_sources]

        if not search_results:
            return "No relevant information found."

        search_results.sort(key=lambda x: x["content"]["metadata"]["start"])

        context_lines = []
        source_labels = []
        for res in search_results:
            meta = res["content"]["metadata"]
            title = self._get_video_title(meta['source'])
            timestamp = self._format_time(meta['start'])

            context_lines.append(f"[{title} @ {timestamp}]: {res['content']['text']}")
            source_labels.append(f"{title} (at {timestamp})")

        full_context = "\n".join(context_lines)

        # POINT 6: Guardrail against Prompt Injection/Hallucinations
        system_msg = f"""You are a precise AI assistant named Lumina. 
        - You must answer ONLY using the provided context.
        - If the answer is not in the context (e.g., about pets/hamsters), politely state you don't know based on these videos.
        - You MUST respond in {target_lang}.
        - Format: Use [Title @ MM:SS] for citations."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {question}"}
                ],
                model=model_name,
                temperature=0
            )

            answer = chat_completion.choices[0].message.content

            # Point 2: Clean Metadata formatting for the UI
            unique_sources = sorted(list(set(source_labels)))
            sources_list = "\n".join([f"1. {s}" for s in unique_sources])

            # Note: The "Verified Evidence" header language is now passed from app.py logic
            return {
                "answer": answer,
                "sources": sources_list
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": ""}

    def translate_text(self, text: str, target_lang: str):
        prompt = f"Translate the following text to {target_lang}. Return ONLY the translation:\n{text}"
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        return completion.choices[0].message.content