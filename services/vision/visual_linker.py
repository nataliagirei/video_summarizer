import re
from pathlib import Path


class VisualLinker:
    """
    Core engine for 'Visual Anchoring'.
    Maps audio segments to the most relevant video frames and
    enriches them with AI-generated visual descriptions.
    """

    def __init__(self, frames_dir, insight_engine):
        """
        Initializes the linker with the frame repository and AI engine.

        Args:
            frames_dir (Path): Path to the directory where frames are stored.
            insight_engine (VideoInsight): The Gemini-powered engine for visual analysis.
        """
        self.frames_dir = Path(frames_dir)
        self.insight_engine = insight_engine

    def _extract_seconds_from_filename(self, filename):
        """
        Parses the standardized filename (e.g., 'frame_0-01-15.jpg')
        to calculate the exact second in the video timeline.

        Returns:
            int: Total seconds from the start of the video.
        """
        match = re.search(r'frame_(\d+)-(\d+)-(\d+)', filename)
        if match:
            h, m, s = map(int, match.groups())
            return h * 3600 + m * 60 + s
        return 0

    def get_anchored_frames(self, segments, detected_lang="en"):
        """
        Synchronizes transcript segments with visual evidence.

        Args:
            segments (list): List of dictionaries containing 'start' and 'text' from Whisper.
            detected_lang (str): Language code or name to ensure Gemini
                                 describes images in the video's native language.

        Returns:
            list: Data packets containing audio, frame path, and AI visual context.
        """
        # 1. Retrieve and index all available frames chronologically
        frame_files = sorted(self.frames_dir.glob("*.jpg"))
        frames_with_time = [
            {"path": str(f), "time": self._extract_seconds_from_filename(f.name)}
            for f in frame_files
        ]

        anchored_data = []

        # 2. Iterate through transcript segments to find the best visual match
        for seg in segments:
            seg_start = seg['start']
            audio_text = seg['text']

            # Locate the frame that appeared most recently before or at the segment start
            best_frame = None
            for f in frames_with_time:
                if f['time'] <= seg_start:
                    best_frame = f['path']
                else:
                    # Frames are sorted; if we exceed the start time, stop searching
                    break

            # 3. Multimodal Analysis: Request a BRIEF description from Gemini
            # We pass the detected language to keep the audit report consistent.
            video_context = ""
            if best_frame:
                video_context = self.insight_engine.describe_frame(
                    frame_path=best_frame,
                    context_text=audio_text,
                    target_lang=detected_lang
                )

            # 4. Compile the evidence packet
            anchored_data.append({
                "start": seg_start,
                "audio_track": audio_text,
                "frame": best_frame,
                "video_context": video_context
            })

        return anchored_data
