"""
Voice I/O
  TTS : edge-tts  (Microsoft neural TTS — free, no API key, natural voice)
  STT : faster-whisper  (local Whisper — free, runs on CPU/GPU)
"""
import sys
import os
import asyncio
import tempfile
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import edge_tts
import soundfile as sf
from faster_whisper import WhisperModel


# ─── TTS ──────────────────────────────────────────────────────────────────────

class TextToSpeech:
    """
    Neural TTS via Microsoft Edge (edge-tts).
    Voices: en-US-GuyNeural (male), en-US-JennyNeural (female), etc.
    Completely free — uses Edge's cloud endpoint, no API key needed.
    """

    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+0%", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume

    def speak(self, text: str):
        """Blocking: synthesise text and play through speakers"""
        asyncio.run(self._speak_async(text))

    async def _speak_async(self, text: str):
        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate, volume=self.volume)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await communicate.save(tmp_path)
            data, samplerate = sf.read(tmp_path, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
        finally:
            os.unlink(tmp_path)

    def available_voices(self):
        return [
            "en-US-GuyNeural", "en-US-JennyNeural",
            "en-US-AriaNeural", "en-GB-RyanNeural",
            "en-IN-NeerjaNeural", "en-AU-NatashaNeural"
        ]


# ─── STT ──────────────────────────────────────────────────────────────────────

class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        model_size: tiny | base | small | medium | large-v3
        base  → fast, good accuracy, ~150MB
        small → better accuracy, ~500MB
        """
        print(f"[STT] Loading Whisper '{model_size}' model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT] Model ready.")

    def record(self, duration: int = 30, samplerate: int = 16000) -> np.ndarray:
        """
        Record from mic for up to `duration` seconds.
        Stops early if silence detected after speech begins.
        Returns numpy float32 array.
        """
        print(f"\n[MIC] Recording (max {duration}s) — speak now, press Enter to stop early...")

        frames = []
        stop_event = threading.Event()

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        # Allow Enter key to stop recording early
        def wait_for_enter():
            input()
            stop_event.set()

        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()

        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback):
            for _ in range(int(duration * 10)):  # check every 100ms
                if stop_event.is_set():
                    break
                sd.sleep(100)

        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        return audio

    def transcribe(self, audio: np.ndarray, samplerate: int = 16000) -> str:
        """Transcribe audio array to text using Whisper"""
        if audio is None or len(audio) == 0:
            return ""

        # Write to temp WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(tmp_path, samplerate, audio_int16)

        try:
            segments, info = self.model.transcribe(tmp_path, language="en", beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        finally:
            os.unlink(tmp_path)

    def listen(self, duration: int = 60) -> str:
        """Record and transcribe in one call"""
        audio = self.record(duration=duration)
        if len(audio) == 0:
            return ""
        print("[STT] Transcribing...")
        text = self.transcribe(audio)
        return text
