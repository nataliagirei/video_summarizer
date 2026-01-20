from pathlib import Path
import sys

from ingestion.youtube_downloader import YouTubeDownloader
from ingestion.audio_extractor import AudioExtractor
from asr.whisper_transcriber import WhisperTranscriber


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <youtube_url>")
        sys.exit(1)

    youtube_url = sys.argv[1]

    print("Starting video ingestion...")

    # 1) Download video
    raw_dir = Path("data/raw")
    downloader = YouTubeDownloader(raw_dir)

    try:
        metadata = downloader.download(youtube_url)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print("Download completed")
    print("Video metadata:")
    print(f"  - Title: {metadata['title']}")
    print(f"  - Duration: {metadata['duration']} seconds")
    print(f"  - File: {metadata['filepath']}")

    # 2) Extract audio
    print("\nExtracting audio...")
    audio_dir = Path("data/audio")
    extractor = AudioExtractor(audio_dir)

    audio_path = extractor.extract(Path(metadata["filepath"]))
    print(f"Audio saved to: {audio_path}")

    # 3) Transcribe
    print("\nTranscribing audio with Whisper...")
    transcript_dir = Path("data/transcripts")
    transcriber = WhisperTranscriber(transcript_dir, model_name="base")

    transcript = transcriber.transcribe(audio_path)
    print(f"Transcript saved to: {transcript_dir / f'{audio_path.stem}.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
