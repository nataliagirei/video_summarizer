from dataclasses import dataclass
from typing import List, Dict, Any
import tiktoken
from processing.text_clean import TextCleaner


@dataclass
class Chunk:
    start: float
    end: float
    text: str
    vector_text: str
    tokens: int


class TextChunker:
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 150, overlap_tokens: int = 20):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.cleaner = TextCleaner()

    def chunk_by_segments(self, segments: List[Dict[str, Any]], lang: str) -> List[Chunk]:
        if not segments:
            return []

        chunks: List[Chunk] = []
        i = 0
        num_segments = len(segments)

        while i < num_segments:
            current_chunk_segments = []
            current_tokens = 0

            # 1. Form the chunk
            for j in range(i, num_segments):
                seg_text = segments[j]["text"]
                seg_tokens = len(self.encoder.encode(seg_text))

                # Handle exceptionally large segments
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

            # 2. Store the result
            raw_text = " ".join([s["text"] for s in current_chunk_segments]).strip()
            vector_text = self.cleaner.clean_for_vector(raw_text, lang)

            chunks.append(Chunk(
                start=current_chunk_segments[0]["start"],
                end=current_chunk_segments[-1]["end"],
                text=raw_text,
                vector_text=vector_text,
                tokens=current_tokens
            ))

            # 3. Calculate step (index i for the next chunk)
            # Find the rollback point for overlap_tokens
            next_start_index = i + len(current_chunk_segments)

            if next_start_index >= num_segments:
                break

            # Perform rollback for overlap
            overlap_accumulated_tokens = 0
            rev_idx = next_start_index - 1

            # Backtrack from the end of the current chunk until overlap_tokens is reached
            while rev_idx > i:
                seg_tokens = len(self.encoder.encode(segments[rev_idx]["text"]))
                if overlap_accumulated_tokens + seg_tokens > self.overlap_tokens:
                    break
                overlap_accumulated_tokens += seg_tokens
                rev_idx -= 1

            # Set i for the next chunk, accounting for overlap
            # Ensure we always move forward by at least one segment
            i = max(rev_idx + 1, i + 1)

        return chunks