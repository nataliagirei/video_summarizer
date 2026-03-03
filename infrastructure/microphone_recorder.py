import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time


class AudioRecorder:
    def __init__(self, output_dir: Path, sample_rate: int = 16000):
        """
        Initialize audio recorder.
        :param output_dir: Directory where recordings will be saved.
        :param sample_rate: Sampling rate for recording (default 16kHz for Whisper ASR).
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fs = sample_rate

    def record(self, duration: float = 10) -> Path | None:
        """
        Records audio for the given duration and saves it as a WAV file.
        Returns the Path to the saved file or None on failure.
        """
        filename = f"record_{int(time.time())}.wav"
        filepath = self.output_dir / filename

        try:
            # Use PCM_16 for maximum compatibility
            with sf.SoundFile(str(filepath), mode='x', samplerate=self.fs,
                              channels=1, subtype='PCM_16') as file:
                try:
                    with sd.InputStream(samplerate=self.fs, channels=1,
                                        callback=self._make_callback(file)):
                        sd.sleep(int(duration * 1000))
                except Exception as stream_error:
                    print(f"InputStream error during recording: {stream_error}")
                    return None

            return filepath

        except Exception as file_error:
            # Handles file creation errors, e.g., permission denied or path issues
            print(f"Recording Error: {file_error}")
            return None

    def _make_callback(self, file):
        """Generates a callback to write audio blocks to file in real-time."""
        def callback(indata, frames, time_info, status):
            if status:
                # Important for debugging input overflows or device issues
                print(f"Stream Status: {status}")
            try:
                file.write(indata.copy())
            except Exception as write_error:
                print(f"Error writing audio block: {write_error}")

        return callback