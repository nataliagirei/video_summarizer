import json
import os
import time
from typing import List, Dict

from groq import Groq


class Translator:
    """
    Handles on-demand text translation using Groq Cloud API.
    Optimized for high accuracy and consistency across different source languages.

    Features:
    - Multi-stage prompt validation to prevent English language 'leakage'.
    - Persistent local caching to minimize API usage and costs.
    - Batch processing to stay within Groq API rate limits (RPM/TPM).
    """

    def __init__(self, cache_dir="data/processed"):
        """Initializes the Groq client and loads the persistent translation cache."""
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "translation_cache.json")

        # Load existing translations from disk to avoid re-translating same strings
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                try:
                    self.translations = json.load(f)
                except json.JSONDecodeError:
                    self.translations = {}
        else:
            self.translations = {}

        # API Setup
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is missing")

        self.client = Groq(api_key=groq_key)
        self.model_name = "llama-3.3-70b-versatile"

    def translate_segments(self, segments: List[Dict], target_lang: str, batch_size: int = 15) -> List[Dict]:
        """
        Translates transcript segments in batches.
        Uses a caching mechanism to bypass API calls for known phrases.
        """
        if not segments:
            return []

        final_results = [None] * len(segments)
        to_translate_indices = []
        texts_to_translate = []

        # 1. Check cache first
        for idx, seg in enumerate(segments):
            original_text = seg.get('text', '')
            # Create a unique key based on text and destination language
            cache_key = f"{original_text}||{target_lang}"

            if cache_key in self.translations:
                new_seg = seg.copy()
                new_seg['text'] = self.translations[cache_key]
                final_results[idx] = new_seg
            else:
                to_translate_indices.append(idx)
                texts_to_translate.append(original_text)

        # 2. Translate only missing segments
        if texts_to_translate:
            for i in range(0, len(texts_to_translate), batch_size):
                current_batch_texts = texts_to_translate[i: i + batch_size]
                current_batch_indices = to_translate_indices[i: i + batch_size]

                translated_list = self._request_batch_translation(current_batch_texts, target_lang)

                for idx_in_batch, translated_text in enumerate(translated_list):
                    original_idx = current_batch_indices[idx_in_batch]
                    original_text = texts_to_translate[i + idx_in_batch]

                    new_seg = segments[original_idx].copy()
                    new_seg['text'] = translated_text
                    final_results[original_idx] = new_seg

                    # Update persistent cache
                    self.translations[f"{original_text}||{target_lang}"] = translated_text

                # Grace period to avoid hitting burst rate limits
                if len(texts_to_translate) > batch_size:
                    time.sleep(1.2)

        self._save_cache()
        return final_results

    def _request_batch_translation(self, texts: List[str], target_lang: str) -> List[str]:
        """
        Sends a batch of strings to the LLM with strict translation instructions.
        Forces the output to be strictly in the target language.
        """
        # Strict System Prompt to prevent mixing languages
        system_prompt = (
            f"You are a professional, high-fidelity translator. "
            f"Translate the following JSON list of strings into {target_lang}. "
            f"CRITICAL RULES: \n"
            f"1. Every single string MUST be translated into {target_lang}.\n"
            f"2. DO NOT leave words in English, even if they are fashion terms or technical jargon.\n"
            f"3. Maintain the exact same count of items in the resulting list.\n"
            f"4. Respond ONLY with a valid JSON object containing the key 'translations'."
        )

        user_payload = json.dumps({"texts": texts}, ensure_ascii=False)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload}
                ],
                temperature=0,  # Zero temperature ensures deterministic and accurate translation
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content
            parsed_res = json.loads(raw_content)
            translated_list = parsed_res.get("translations", [])

            if len(translated_list) != len(texts):
                print(f"⚠️ Mismatch: Sent {len(texts)}, Got {len(translated_list)}")
                return texts

            return translated_list

        except Exception as e:
            print(f"⚠️ Translation API Error: {e}")
            return texts

    def _save_cache(self):
        """Saves translation memory to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.translations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Cache Save Error: {e}")
