from dataclasses import dataclass
from typing import List, Dict, Any
import tiktoken
from services.processing.text_clean import TextCleaner


@dataclass
class Chunk:
    """Represents a processed text segment for the knowledge base."""
    start: float
    end: float
    text: str          # Context for LLM (with timestamps)
    vector_text: str   # Clean text for vector embedding (FAISS)
    tokens: int


class TextChunker:
    """
    Handles transcript segmentation into optimal blocks for RAG.
    Preserves temporal alignment by injecting timestamps directly into the text context.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 150, overlap_tokens: int = 20):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.cleaner = TextCleaner()

    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to [MM:SS] format for LLM readability."""
        m, s = divmod(int(seconds), 60)
        return f"[{m:02d}:{s:02d}]"

    def chunk_by_segments(self, segments: List[Dict[str, Any]], lang: str) -> List[Chunk]:
        """
        Groups transcript segments into chunks while preserving time-alignment.
        Each segment is prefixed with a timestamp to prevent hallucinations.
        """
        if not segments:
            return []

        chunks: List[Chunk] = []
        i = 0
        num_segments = len(segments)

        while i < num_segments:
            current_chunk_segments = []
            current_tokens = 0

            # --- Accumulate segments until max_tokens limit ---
            for j in range(i, num_segments):
                seg_text = segments[j]["text"]
                seg_tokens = len(self.encoder.encode(seg_text))

                # If a single segment is larger than max_tokens, force it into a chunk
                if seg_tokens > self.max_tokens and not current_chunk_segments:
                    current_chunk_segments.append(segments[j])
                    current_tokens = seg_tokens
                    break

                if current_tokens + seg_tokens > self.max_tokens:
                    break

                current_chunk_segments.append(segments[j])
                current_tokens += seg_tokens

            if not current_chunk_segments:
                break

            # --- Prepare text for LLM and vector search ---
            raw_text_with_timestamps = " ".join(
                f"{self.format_time(s['start'])} {s['text']}" for s in current_chunk_segments
            ).strip()

            vector_text_source = " ".join(s["text"] for s in current_chunk_segments).strip()
            vector_text = self.cleaner.clean_for_vector(vector_text_source, lang)

            chunks.append(Chunk(
                start=current_chunk_segments[0]["start"],
                end=current_chunk_segments[-1]["end"],
                text=raw_text_with_timestamps,
                vector_text=vector_text,
                tokens=current_tokens
            ))

            # --- Determine next start index with overlap ---
            next_start_index = i + len(current_chunk_segments)
            if next_start_index >= num_segments:
                break

            # Compute overlap tokens backwards
            overlap_accumulated_tokens = 0
            rev_idx = next_start_index - 1
            while rev_idx > i:
                seg_tokens = len(self.encoder.encode(segments[rev_idx]["text"]))
                if overlap_accumulated_tokens + seg_tokens > self.overlap_tokens:
                    break
                overlap_accumulated_tokens += seg_tokens
                rev_idx -= 1

            # Move i forward, at least by one segment
            i = max(rev_idx + 1, i + 1)

        return chunks