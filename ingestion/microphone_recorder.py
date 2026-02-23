import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import time


class AudioRecorder:
    def __init__(self, output_dir: Path):
        # Targeting data/audio or data/raw
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fs = 16000  # Optimal sample rate for Whisper ASR

    def record(self, duration=10):
        """
        Records audio as a stream directly to the disk.
        Optimized for memory efficiency during long audit sessions.
        """
        filename = f"record_{int(time.time())}.wav"
        filepath = self.output_dir / filename

        # Using PCM_16 subtype ensures maximum compatibility with Whisper
        try:
            with sf.SoundFile(str(filepath), mode='x', samplerate=self.fs,
                              channels=1, subtype='PCM_16') as file:
                with sd.InputStream(samplerate=self.fs, channels=1,
                                    callback=self._make_callback(file)):
                    sd.sleep(int(duration * 1000))

            return filepath

        except Exception as e:
            # Important for debugging in Streamlit
            print(f"Recording Error: {e}")
            return None

    def _make_callback(self, file):
        """Writes received audio blocks to file in real-time."""
        def callback(indata, frames, time, status):
            if status:
                # Log potential issues like input overflow (important for data integrity)
                print(f"Status: {status}")
            file.write(indata.copy())
        return callback