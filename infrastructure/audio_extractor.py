import subprocess
from pathlib import Path


class AudioExtractor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: Path) -> Path | None:
        """
        Extracts audio from a video file to WAV format (16kHz, mono).
        Returns Path to WAV file or None if extraction failed.
        """
        audio_path = self.output_dir / f"{video_path.stem}.wav"

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
            "-y"
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"Audio extraction failed for {video_path}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during audio extraction: {e}")
            return None
