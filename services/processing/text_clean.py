import re


class TextCleaner:
    """
    Cleans transcript text specifically for vector embeddings.
    Removes only undisputed filler words while keeping meaningful context.
    """

    def __init__(self):
        # Filler words per language to remove for vector search
        self.fillers = {
            "ru": ["ну", "э-э", "м-м", "а-а", "типа", "вот"],
            "en": ["uhm", "uh", "um", "er", "ah"],
            "pl": ["y-y", "e-e", "no", "mmm"],
            "de": ["äh", "ähm"],
            "fr": ["euh", "beh"],
            "es": ["eh", "em", "pues", "o sea"],
            "ko": ["어", "음", "그니까", "저기"],
            "zh": ["嗯", "那个", "然后", "呃"],
            "ja": ["えーと", "あの", "ま", "うん"],
            "vi": ["à", "ừm", "thì", "là"],
        }

    def clean_for_vector(self, text: str, lang: str) -> str:
        """
        Cleans text from filler sounds to improve vector search precision.
        Keeps all words that could carry semantic meaning.

        Args:
            text (str): original transcript text
            lang (str): language code (e.g., "en", "ru")

        Returns:
            str: cleaned text ready for vector embeddings
        """
        if not text:
            return ""

        # Lowercase for consistency
        text_lower = text.lower()

        # Remove repeated transcription artifacts like "....." or "!!!!!!"
        text_lower = re.sub(r'(.)\1{4,}', r'\1', text_lower)

        words = text_lower.split()
        stop_list = self.fillers.get(lang, [])

        # If language unknown, return the lowercased text as-is
        if not stop_list:
            return text_lower

        # Remove only confirmed fillers
        cleaned = [w for w in words if w.strip(",.?!:;()") not in stop_list]

        return " ".join(cleaned)
