import re


class TextCleaner:
    def __init__(self):
        # Only undisputed filler sounds and interjections are kept.
        # Words that could carry meaning (like "actually", "anyway", "like") are REMOVED
        # from these lists to preserve audit integrity.
        self.fillers = {
            "ru": ["ну", "э-э", "м-м", "а-а", "типа", "вот"],
            "en": ["uhm", "uh", "um", "er", "ah"],
            "pl": ["y-y", "e-e", "no", "mmm"],
            "de": ["äh", "ähm"],
            "fr": ["euh", "beh"],
            "es": ["eh", "em", "pues", "o sea"],  # Spanish: "o sea" is a very common filler
            "ko": ["어", "음", "그니까", "저기"],  # Korean: "eo", "eum", "geunikka", "jeogi" (fillers)
            "zh": ["嗯", "那个", "然后", "呃"],  # Chinese: "en", "na-ge", "ran-hou", "e"
            "ja": ["えーと", "あの", "ま", "うん"],  # Japanese: "eto", "ano", "ma", "un"
            "vi": ["à", "ừm", "thì", "là"],  # Vietnamese: "à", "ừm", "thì", "là"
        }

    def clean_for_vector(self, text: str, lang: str) -> str:
        """
        Cleans text ONLY from filler sounds to improve vector search precision.
        Maintains all words that could potentially carry semantic meaning.
        """
        if not text:
            return ""

        # Standardize: lowercase for the vector engine
        text_lower = text.lower()

        # Remove transcription artifacts (Whisper loops like "........")
        text_lower = re.sub(r'(.)\1{4,}', r'\1', text_lower)

        words = text_lower.split()
        stop_list = self.fillers.get(lang, [])

        # If the language is unknown, we don't remove anything (Safety First)
        if not stop_list:
            return text_lower

        # Filter out only confirmed fillers
        cleaned = [w for w in words if w.strip(",.?!:;()") not in stop_list]
        return " ".join(cleaned)