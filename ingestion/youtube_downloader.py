import yt_dlp
from pathlib import Path
from tqdm import tqdm
import json
import os


class YouTubeDownloader:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Path to the centralized metadata store
        self.registry_path = self.output_dir / "registry.json"
        self.pbar = None

    def _progress_hook(self, d):
        """Hook to update the terminal progress bar during download."""
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if self.pbar is None and total:
                self.pbar = tqdm(total=total, unit="B", unit_scale=True)
            if self.pbar:
                self.pbar.n = d.get("downloaded_bytes", 0)
                self.pbar.refresh()
        elif d["status"] == "finished" and self.pbar:
            self.pbar.close()
            self.pbar = None

    def _update_registry(self, metadata: dict):
        """
        Updates the centralized registry.json file.
        Loads existing data, appends/updates the new video entry, and saves.
        """
        registry = {}

        # Load existing registry if it exists
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except json.JSONDecodeError:
                # Handle corrupted or empty JSON files
                registry = {}

        # Add or update entry using video_id as the unique key
        registry[metadata["video_id"]] = metadata

        # Save updated registry with pretty formatting
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def download(self, url: str) -> dict:
        """
        Downloads audio from YouTube and returns metadata.
        Uses yt-dlp with custom headers to prevent 403 Forbidden errors.
        """
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "progress_hooks": [self._progress_hook],
            "quiet": False,
            "no_warnings": False,
            "nocheckcertificate": True,
            "add_header": [
                'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language: en-US,en;q=0.5',
                'Sec-Fetch-Mode: navigate'
            ],
            "referer": "https://www.google.com/",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        metadata = {
            "video_id": info["id"],
            "title": info["title"],
            "duration": info["duration"],
            "filepath": str(self.output_dir / f"{info['id']}.{info['ext']}")
        }

        # Instead of multiple JSONs, update the single registry
        self._update_registry(metadata)

        return metadata