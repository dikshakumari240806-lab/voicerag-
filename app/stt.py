import os
import tempfile

from faster_whisper import WhisperModel
from .config import settings


class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8"
        )

    def transcribe_file(self, path):
        segments, info = self.model.transcribe(
            path,
            language="en",
            vad_filter=True,
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False
        )

        text = " ".join(
            s.text.strip()
            for s in segments
            if s.text.strip()
        ).strip()

        return text, float(
            getattr(info, "language_probability", 1.0)
        )


def record_wav(seconds=6):
    import sounddevice as sd
    import soundfile as sf

    SAMPLE_RATE = 44100
    DEVICE = 12

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    print(f"Using microphone device: {DEVICE}")
    print(f"Recording {seconds} seconds — speak now...")

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=DEVICE
    )

    sd.wait()

    # Reduce microphone gain
    audio = audio * 0.35

    # Keep audio safely inside [-1, 1]
    audio = audio.clip(-1.0, 1.0)

    print(f"Audio peak: {abs(audio).max():.4f}")
    print(f"Audio RMS:  {(audio ** 2).mean() ** 0.5:.4f}")

    sf.write(path, audio, SAMPLE_RATE)

    return path