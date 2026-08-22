import os
import tempfile
from faster_whisper import WhisperModel

_model = None


def get_model():
    global _model

    if _model is None:
        _model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1
        )

    return _model


def transcribe_file(path):
    model = get_model()

    segments, info = model.transcribe(
        path,
        language="en",
        beam_size=1,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    return text, float(
        getattr(info, "language_probability", 1.0)
    )


class Transcriber:

    def __init__(self):
        self.model = get_model()

    def transcribe_file(self, path):
        return transcribe_file(path)


def record_wav(seconds=6):
    import sounddevice as sd
    import soundfile as sf

    sample_rate = 16000

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        path,
        audio,
        sample_rate
    )

    return path