def speak(text):
    try:
        import pyttsx3
        e=pyttsx3.init(); e.say(text); e.runAndWait()
    except Exception as exc: print("TTS unavailable:",exc)
