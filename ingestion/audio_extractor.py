from pathlib import Path
import subprocess


class AudioExtractor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: Path) -> Path:
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

        subprocess.run(cmd, check=True)
        return audio_path
