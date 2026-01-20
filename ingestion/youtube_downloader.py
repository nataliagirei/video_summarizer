import yt_dlp
from pathlib import Path
from tqdm import tqdm
import json


class YouTubeDownloader:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pbar = None

    def _progress_hook(self, d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if self.pbar is None and total:
                self.pbar = tqdm(total=total, unit="B", unit_scale=True)

            if self.pbar:
                self.pbar.n = d.get("downloaded_bytes", 0)
                self.pbar.refresh()

        elif d["status"] == "finished" and self.pbar:
            self.pbar.close()
            self.pbar = None  # важно сбрасывать

    def download(self, url: str) -> dict:
        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "progress_hooks": [self._progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        metadata = {
            "video_id": info["id"],
            "title": info["title"],
            "duration": info["duration"],
            "filepath": str(self.output_dir / f"{info['id']}.{info['ext']}")
        }

        with open(self.output_dir / f"{info['id']}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

