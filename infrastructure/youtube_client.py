import json
import os
from pathlib import Path

import yt_dlp
from tqdm import tqdm


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
        """Updates the centralized registry.json file."""
        registry = {}

        # Load existing registry if it exists
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except json.JSONDecodeError:
                registry = {}

        # Add or update entry using video_id as the unique key
        registry[metadata["video_id"]] = metadata

        # Save updated registry
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def download(self, url: str) -> dict:
        """
        Downloads video and audio from YouTube and returns metadata.
        Safe against unavailable videos or network issues.
        """
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
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

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info and download
                info = ydl.extract_info(url, download=True)
                # Get the actual filename after merging video and audio
                final_filename = ydl.prepare_filename(info)
                # Ensure the extension matches the merged format (mp4)
                if not final_filename.endswith(".mp4"):
                    final_filename = os.path.splitext(final_filename)[0] + ".mp4"

            metadata = {
                "video_id": info["id"],
                "title": info["title"],
                "duration": info["duration"],
                "video_url": info.get("webpage_url"),
                "filepath": final_filename
            }

            # Update the single registry
            self._update_registry(metadata)
            return metadata

        except Exception as e:
            # Safe failure, log error and return empty dict
            print(f"Error downloading YouTube video ({url}): {e}")
            return {}
