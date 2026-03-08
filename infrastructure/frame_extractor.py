import os
from datetime import timedelta
from pathlib import Path

import cv2


class FrameProcessor:
    """
    Module for extracting and analyzing video frames to provide
    visual context for audit reports.
    """

    def __init__(self, output_dir):
        """
        Initializes the processor and ensures the output directory exists.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path, interval_seconds=10):
        """
        Extracts frames from a video file at specified intervals using fast seeking.
        Returns a list of Paths to the saved images.
        """
        frame_paths = []
        video_path_str = str(video_path)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path_str)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path_str}")
            return []

        # Retrieve video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate video file integrity
        if not fps or fps == 0 or total_frames == 0:
            print(f"Error: Could not read video parameters for {video_path_str}")
            cap.release()
            return []

        duration_seconds = total_frames / fps

        # Iterate through time based on the specified interval
        for sec in range(0, int(duration_seconds), interval_seconds):
            # Calculate the specific frame ID for the given second
            frame_id = int(sec * fps)

            # Move the video cursor directly to the frame (Fast Seeking)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Create a readable timestamp for the filename (e.g., 00-01-30)
                timestamp_str = str(timedelta(seconds=sec)).replace(":", "-")
                filename = f"frame_{timestamp_str}.jpg"
                save_path = self.output_dir / filename

                try:
                    # Save the frame as JPEG
                    cv2.imwrite(str(save_path), frame)
                    frame_paths.append(save_path)
                except Exception as e:
                    print(f"Error saving frame {save_path}: {e}")
            else:
                print(f"Warning: Could not read frame at {sec} seconds")
                continue

        cap.release()
        return frame_paths

    def analyze_scene_change(self, frame_paths, threshold=0.8):
        """
        Optional method to filter out nearly identical frames.
        Compares color histograms to detect significant scene changes.
        """
        if not frame_paths:
            return []

        unique_frames = [frame_paths[0]]

        for i in range(1, len(frame_paths)):
            img1 = cv2.imread(str(frame_paths[i - 1]))
            img2 = cv2.imread(str(frame_paths[i]))

            if img1 is None or img2 is None:
                print(f"Warning: Could not read one of the frames {frame_paths[i - 1]}, {frame_paths[i]}")
                continue

            # Calculate histograms for comparison
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

            # Normalize histograms
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)

            # Compare histograms using correlation
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # If similarity is below the threshold, consider it a new scene
            if score < threshold:
                unique_frames.append(frame_paths[i])
            else:
                # Optionally delete the redundant frame from disk to save space
                try:
                    os.remove(str(frame_paths[i]))
                except Exception as e:
                    print(f"Could not remove redundant frame {frame_paths[i]}: {e}")

        return unique_frames
