import os
import sys
from pathlib import Path

# Twoje moduły
from ingestion.youtube_downloader import YouTubeDownloader
from ingestion.audio_extractor import AudioExtractor
from ingestion.microphone_recorder import AudioRecorder
from asr.whisper_transcriber import WhisperTranscriber
from processing.transcript_processor import TranscriptProcessor
from processing.text_chunk import TextChunker
from processing.vector_store import LocalVectorStore

# Fix dla bibliotek na Macu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    print("--- VideoInsight AI: Content Ingestion Pipeline ---")
    print("1) YouTube URL")
    print("2) Microphone Recording")
    print("3) Local File (Video/Audio)")
    choice = input("Select input source (1, 2, 3): ").strip()

    # Struktura folderów zgodna z Twoim PyCharmem
    dirs = {
        "raw": Path("data/raw"),
        "audio": Path("data/audio"),
        "transcripts": Path("data/transcripts"),
        "processed": Path("data/processed")
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    source_id = ""
    audio_path = None
    source_url = "local_access"

    # --- KROK 1: Pozyskanie Audio ---
    if choice == "1":
        source_url = input("Enter YouTube URL: ").strip()
        downloader = YouTubeDownloader(dirs["raw"])
        metadata = downloader.download(source_url)
        source_id = metadata["video_id"]
        audio_path = AudioExtractor(dirs["audio"]).extract(Path(metadata["filepath"]))

    elif choice == "2":
        duration = int(input("Enter recording duration (seconds): ") or 10)
        audio_path = AudioRecorder(dirs["audio"]).record(duration=duration)
        source_id = audio_path.stem
        source_url = "microphone_recording"

    elif choice == "3":
        file_path = Path(input("Enter full path to local file: ").strip().replace("'", "").replace('"', ""))
        if not file_path.exists():
            print("❌ File not found!");
            return

        source_id = file_path.stem
        # Obsługa wideo vs audio
        video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.webm']
        if file_path.suffix.lower() in video_extensions:
            print(f"🎥 Video detected. Extracting audio from {file_path.name}...")
            audio_path = AudioExtractor(dirs["audio"]).extract(file_path)
        else:
            print(f"🎵 Audio file detected: {file_path.name}")
            audio_path = file_path

    else:
        print("Invalid choice.");
        return

    # --- KROK 2: Transkrypcja ---
    print(f"🚀 Transcribing: {audio_path.name}...")
    # Model 'base' jest szybki, ale jeśli masz czas, 'small' będzie dokładniejszy dla audytu
    transcriber = WhisperTranscriber(dirs["transcripts"], model_name="base")
    transcript = transcriber.transcribe(audio_path)

    # --- KROK 3: Przetwarzanie i Chunking (PRECYZJA) ---
    processor = TranscriptProcessor(dirs["processed"])
    processed = processor.process(transcript, source_id)
    lang = transcript.get("language", "en")

    print(f"✂️ Chunking with HIGH PRECISION (max_tokens=150)...")
    # Zmniejszone max_tokens = lepsze trafienia w konkretne sekundy
    chunker = TextChunker(max_tokens=150, overlap_tokens=30)
    chunks = chunker.chunk_by_segments(processed["segments"], lang)

    prepared_chunks = []
    for idx, ch in enumerate(chunks):
        prepared_chunks.append({
            "id": f"{source_id}_{idx}",
            "text": ch.text,
            "vector_text": ch.vector_text,
            "metadata": {
                "source": source_id,
                "url": source_url,
                "start": ch.start,
                "end": ch.end,
                "language": lang
            }
        })

    # --- KROK 4: Baza FAISS ---
    print(f"🧠 Indexing {len(prepared_chunks)} fragments...")
    store = LocalVectorStore()
    store.add_chunks(prepared_chunks)
    store.persist()

    print(f"✅ DONE! Source '{source_id}' is ready for questions in ask_it.py")


if __name__ == "__main__":
    main()