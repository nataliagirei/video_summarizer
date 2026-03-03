import os
import json
from typing import List, Dict
from groq import Groq


class Translator:
    """
    Handles on-demand text translation using Groq (RAG).
    Caches translations locally to avoid duplicate calls.
    """

    def __init__(self, cache_dir="data/processed"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "translation_cache.json")

        # Load existing translations
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
        else:
            self.translations = {}

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not set in environment variables")
        self.client = Groq(api_key=groq_key)

    def translate_text(self, text: str, target_lang: str) -> str:
        """
        Translate a single text string into the target language using Groq.
        Uses cache to prevent duplicate requests.
        """
        key = f"{text}||{target_lang}"
        if key in self.translations:
            return self.translations[key]

        prompt = f"Translate the following text into {target_lang}:\n{text}"
        response = self.client.generate(prompt=prompt, max_output_tokens=500)
        translated = response.text.strip()

        self.translations[key] = translated
        self._save_cache()
        return translated

    def translate_segments(self, segments: List[Dict], target_lang: str) -> List[Dict]:
        """
        Translate a list of transcript segments.
        Each segment must have 'start' and 'text' keys.
        """
        translated_segments = []
        for seg in segments:
            ts = seg.copy()
            ts['text'] = self.translate_text(seg['text'], target_lang)
            translated_segments.append(ts)
        return translated_segments

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.translations, f, ensure_ascii=False, indent=2)