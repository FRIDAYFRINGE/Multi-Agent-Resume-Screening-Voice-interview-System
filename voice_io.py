"""
Voice I/O
  TTS : edge-tts      (Microsoft neural TTS — free, no API key, natural voice)
  STT : AssemblyAISTT (AssemblyAI cloud streaming STT — universal-3-5-pro real-time v3)
       GroqSTT        (Groq cloud Whisper large-v3 — fast cloud Whisper)
       faster-whisper (local Whisper — free, runs on CPU/GPU)

Switch between STT providers via STT_PROVIDER in interview_app.py or environment variable.
"""
import sys
import os
import asyncio
import tempfile
import threading
import numpy as np
import sounddevice as sd
import edge_tts
import soundfile as sf
from typing import Optional, Callable, Dict, Any, List
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
            sf.write(tmp_path, audio_int16, samplerate)

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


# ─── Groq Cloud STT ───────────────────────────────────────────────────────────

class GroqSTT:
    """
    Cloud STT via Groq's Whisper large-v3 API.
    Free tier: 28,800 audio seconds/day (~8 hours).
    ~20x faster than local CPU inference for the same model.
    Requires GROQ_API_KEY in .env  →  https://console.groq.com/keys
    """

    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, model: str = "whisper-large-v3"):
        self.model = model
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set — add it to .env or set STT_PROVIDER='local'")
        print(f"[STT] Groq cloud Whisper '{model}' ready.")

    def transcribe_file(self, audio_path: str, prompt: str = "") -> str:
        """Send audio file to Groq and return transcript text."""
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "language": "en", "response_format": "text"}
        if prompt:
            data["prompt"] = prompt[:224]  # Groq caps prompt at 224 tokens
        with open(audio_path, "rb") as f:
            resp = requests.post(
                self.API_URL, headers=headers,
                files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                data=data, timeout=60,
            )
        resp.raise_for_status()
        return (resp.text or "").strip()


# ─── AssemblyAI Cloud Streaming STT ──────────────────────────────────────────

class AssemblyAISTT:
    """
    Cloud STT via AssemblyAI's real-time streaming API (universal-3-5-pro).
    Streaming WebSocket: wss://streaming.assemblyai.com/v3/ws
    Requires ASSEMBLY_AI_API_KEY (or ASSEMBLYAI_API_KEY) in .env.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        speech_model: str = "universal-3-5-pro",
        sample_rate: int = 16000,
    ):
        self.api_key = api_key or os.getenv("ASSEMBLY_AI_API_KEY") or os.getenv("ASSEMBLYAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("ASSEMBLY_AI_API_KEY not set — add it to .env or set STT_PROVIDER='groq'/'local'")
        self.speech_model = speech_model
        self.sample_rate = sample_rate
        print(f"[STT] AssemblyAI streaming '{speech_model}' ready.")

    def _audio_to_pcm16(self, audio_data, input_sample_rate: int = 16000) -> bytes:
        """Convert audio file path or numpy array to 16-bit mono PCM bytes at self.sample_rate."""
        if isinstance(audio_data, str):
            try:
                data, sr = sf.read(audio_data, dtype="int16")
            except Exception:
                import wave
                with wave.open(audio_data, "rb") as wf:
                    sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
                    data = np.frombuffer(raw, dtype=np.int16)
        elif isinstance(audio_data, np.ndarray):
            data = audio_data
            sr = input_sample_rate
            if data.dtype != np.int16:
                if np.issubdtype(data.dtype, np.floating):
                    data = (data * 32767).astype(np.int16)
                else:
                    data = data.astype(np.int16)
        elif isinstance(audio_data, bytes):
            return audio_data
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

        if data.ndim > 1:
            data = data[:, 0]  # downmix stereo to mono

        if sr != self.sample_rate:
            old_indices = np.linspace(0, len(data), len(data), endpoint=False)
            new_length = int(len(data) * self.sample_rate / sr)
            new_indices = np.linspace(0, len(data), new_length, endpoint=False)
            data = np.interp(new_indices, old_indices, data).astype(np.int16)

        return data.tobytes()

    def transcribe_file(
        self,
        audio_path: str,
        prompt: str = "",
        on_turn: Optional[Callable[[str, bool], None]] = None
    ) -> str:
        """
        Stream audio file to AssemblyAI v3 streaming WebSocket and return transcript.
        Calls optional on_turn(transcript, is_final) callback as turns arrive.
        Falls back to AssemblyAI HTTP Transcriber if WebSocket streaming encounters an error.
        """
        pcm_bytes = self._audio_to_pcm16(audio_path)
        if not pcm_bytes:
            return ""

        try:
            from assemblyai.streaming.v3 import (
                RealTimeTranscriber,
                RealTimeTranscriberOptions,
                RealTimeParameters,
                RealTimeEvents,
                Encoding,
                TurnEvent,
            )

            completed_turns = []
            last_partial = ""

            def handle_turn(client, event: TurnEvent):
                nonlocal last_partial
                text = (event.transcript or "").strip()
                if text:
                    last_partial = text
                    if on_turn:
                        try:
                            on_turn(text, bool(event.end_of_turn))
                        except Exception:
                            pass
                    if event.end_of_turn:
                        completed_turns.append(text)

            client = RealTimeTranscriber(
                RealTimeTranscriberOptions(terminate_timeout=15.0),
                api_key=self.api_key,
            )
            client.on(RealTimeEvents.Turn, handle_turn)

            params = RealTimeParameters(
                speech_model=self.speech_model,
                encoding=Encoding.pcm_s16le,
                sample_rate=self.sample_rate,
                prompt=prompt[:500] if prompt else None,
            )
            client.connect(params)

            # Stream audio in 4KB chunks
            chunk_size = 4096
            for i in range(0, len(pcm_bytes), chunk_size):
                client.stream(pcm_bytes[i:i + chunk_size])

            client.disconnect(terminate=True)

            if completed_turns:
                return " ".join(completed_turns).strip()
            elif last_partial:
                return last_partial.strip()
            return ""

        except Exception as stream_err:
            print(f"[AssemblyAI] Streaming error: {stream_err}. Attempting REST fallback...")
            try:
                import assemblyai as aai
                aai.settings.api_key = self.api_key
                res = aai.Transcriber().transcribe(audio_path)
                if res and res.text:
                    return res.text.strip()
            except Exception as rest_err:
                print(f"[AssemblyAI] REST fallback error: {rest_err}")
            raise stream_err

    def listen(self, duration: int = 60, samplerate: int = 16000) -> str:
        """
        Record from mic and stream to AssemblyAI in real time.
        Stops early when Enter is pressed.
        """
        print(f"\n[MIC] Recording (max {duration}s) — speak now, press Enter to stop early...")

        from assemblyai.streaming.v3 import (
            RealTimeTranscriber,
            RealTimeTranscriberOptions,
            RealTimeParameters,
            RealTimeEvents,
            Encoding,
            TurnEvent,
        )

        completed_turns = []
        last_partial = ""
        stop_event = threading.Event()

        def handle_turn(client, event: TurnEvent):
            nonlocal last_partial
            text = (event.transcript or "").strip()
            if text:
                last_partial = text
                if event.end_of_turn:
                    completed_turns.append(text)
                    print(f"\n[STT Turn]: {text}")
                else:
                    print(f"\r[STT ...]: {text}", end="", flush=True)

        client = RealTimeTranscriber(
            RealTimeTranscriberOptions(terminate_timeout=15.0),
            api_key=self.api_key,
        )
        client.on(RealTimeEvents.Turn, handle_turn)

        client.connect(RealTimeParameters(
            speech_model=self.speech_model,
            encoding=Encoding.pcm_s16le,
            sample_rate=self.sample_rate,
        ))

        def wait_for_enter():
            input()
            stop_event.set()

        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()

        def callback(indata, frame_count, time_info, status):
            if not stop_event.is_set():
                audio_int16 = (indata * 32767).astype(np.int16)
                client.stream(audio_int16.tobytes())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback):
            for _ in range(int(duration * 10)):
                if stop_event.is_set():
                    break
                sd.sleep(100)

        client.disconnect(terminate=True)
        print()

        if completed_turns:
            return " ".join(completed_turns).strip()
        elif last_partial:
            return last_partial.strip()
        return ""
