import os
import json
import re
from pathlib import Path
from PIL import Image
from google import genai
from groq import Groq
from dotenv import load_dotenv
from infrastructure.vector_store import LocalVectorStore

# --- CRITICAL MAC STABILITY FIX ---
# Prevents the application from crashing due to OpenMP runtime conflicts
# between FAISS and SentenceTransformers on macOS.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load API keys from .env file automatically
load_dotenv()


class VideoInsight:
    """
    Core AI engine for Lumina handling:
    - RAG (Retrieval-Augmented Generation) via Groq (Llama models).
    - Computer Vision / Frame description via Google Gemini.
    - Automated Audit analysis (MoM, Sentiment, Entities).
    """

    def __init__(self, persist_dir: str = None):
        """Initializes API clients and the vector store."""
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        # Soft check for API keys
        if not self.groq_key:
            print("⚠️ Warning: GROQ_API_KEY not found in environment.")
        if not self.gemini_key:
            print("⚠️ Warning: GEMINI_API_KEY not found in environment.")

        # Initialize the LocalVectorStore (Handles FAISS and Embeddings)
        try:
            self.store = LocalVectorStore(persist_dir=persist_dir)
        except Exception as e:
            print(f"❌ Failed to initialize Vector Store: {e}")
            self.store = None

        # Setup Clients
        self.client = Groq(api_key=self.groq_key) if self.groq_key else None

        if self.gemini_key:
            self.genai_client = genai.Client(api_key=self.gemini_key)
        else:
            self.genai_client = None

        self.vision_model_name = "gemini-2.0-flash"

    def describe_frame(self, frame_path: str, context_text: str, target_lang: str = "en") -> str:
        """
        Analyze a video frame using Gemini Vision.
        Forces the description into the target language.
        """
        if not self.genai_client:
            return "Vision error: Gemini API key missing."

        img = None
        try:
            img = Image.open(frame_path)
            # Strict language instruction in the vision prompt
            prompt = f"""
            Identify key visual evidence in this frame.
            Context from transcript: {context_text}
            Respond STRICTLY in {target_lang}. One concise sentence.
            """
            response = self.genai_client.models.generate_content(
                model=self.vision_model_name,
                contents=[prompt, img]
            )
            return response.text.strip()
        except Exception as e:
            return f"Vision error: {str(e)}"
        finally:
            if img:
                img.close()

    def ask(self, question: str, filter_sources: list = None, target_lang: str = "English",
            model_name: str = "llama-3.3-70b-versatile") -> dict:
        """
        Standard RAG Chat query with strict language protocol.
        """
        if not self.client or not self.store:
            return {"answer": "Error: Engine not fully initialized.", "sources": ""}

        sanitized_question = question.strip()
        search_results = self.store.search(sanitized_question, k=15)

        if filter_sources:
            search_results = [
                r for r in search_results
                if r["content"]["metadata"]["source"] in filter_sources
            ]

        if not search_results:
            return {"answer": "No relevant data found.", "sources": ""}

        search_results.sort(key=lambda x: x["content"]["metadata"]["start"])

        context_lines = [
            f"<{self._format_time(res['content']['metadata']['start'])}>: {res['content']['text']}"
            for res in search_results
        ]
        full_context = "\n".join(context_lines)

        # Language is enforced in the system message
        system_msg = f"""You are Lumina, a secure Audit system.
        STRICT RULES:
        1. Respond ONLY in {target_lang}.
        2. Use ONLY the provided [VIDEO CONTEXT].
        3. Use [MM:SS] timestamps for citations.
        """

        user_content = f"[CONTEXT]\n{full_context}\n\n[USER QUESTION]: {sanitized_question}"

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                model=model_name,
                temperature=0
            )
            answer = chat_completion.choices[0].message.content
            clean_answer = re.sub(r"\[(\d{2}:\d{2})\]", r"**[\1]**", answer)
            return {"answer": clean_answer, "sources": "Verified Audit Records"}
        except Exception as e:
            return {"answer": f"API Error: {str(e)}", "sources": ""}

    def analyze_audit_details(self, source_id: str, target_lang: str = "English") -> str:
        """Audit specific analysis with dynamic language selection."""
        instruction = "Analyze Sentiment, Key Actions, and Entities (Names/Dates)."
        return self._run_analysis(instruction, source_id, target_lang)

    def generate_mom(self, source_id: str, target_lang: str = "English") -> str:
        """MoM generation with dynamic language selection."""
        instruction = "Generate professional, highly structured Minutes of Meeting (MoM)."
        return self._run_analysis(instruction, source_id, target_lang)

    def _run_analysis(self, instruction: str, source_id: str, target_lang: str) -> str:
        """
        Internal helper with forced language mapping to ensure consistency.
        """
        if not self.client:
            return "Error: API client not initialized."

        # Map UI codes to full names for the LLM prompt
        lang_map = {
            "PL": "Polish (Język Polski)",
            "RU": "Russian (Język Rosyjski)",
            "EN": "English"
        }
        full_lang_name = lang_map.get(target_lang, "English")

        try:
            json_path = Path(f"data/processed/{source_id}_processed.json")
            if not json_path.exists():
                return f"Error: Data for source {source_id} not found."

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            context = " ".join([s["text"] for s in data["segments"]])[:25000]

            # REINFORCED SYSTEM PROMPT
            # We explicitly tell the AI to ignore the language of the transcript.
            system_prompt = (
                f"You are a professional auditor. "
                f"STRICT INSTRUCTION: Your output MUST BE entirely in {full_lang_name}. "
                f"Even if the transcript below is in another language, DO NOT use that language. "
                f"Translate all findings and headers into {full_lang_name}."
            )

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{instruction}\n\nTRANSCRIPT CONTEXT:\n{context}"}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Analysis Error: {str(e)}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"