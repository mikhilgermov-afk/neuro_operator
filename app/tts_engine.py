import asyncio
import base64
import io
import os
import wave
import numpy as np
import requests


class TTSEngine:
    def __init__(self, ref_audio_path: str = "", device: str = "cuda"):
        base = os.getenv("TTS_API_URL", "http://tts-service:10002").rstrip("/")
        if base.endswith("/tts") or base.endswith("/synthesize"):
            self.endpoints = [base]
        else:
            self.endpoints = [base + "/tts", base + "/synthesize"]
        self.timeout = float(os.getenv("TTS_TIMEOUT", "120"))

    async def generate(self, text: str):
        last_exc = RuntimeError("TTS request failed")
        for url in self.endpoints:
            try:
                return await asyncio.to_thread(self._request, url, text)
            except Exception as e:
                last_exc = e
        raise last_exc

    def _request(self, url: str, text: str):
        r = requests.post(url, json={"text": text}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"TTS HTTP {r.status_code}: {r.text[:500]}")

        ct = (r.headers.get("content-type") or "").lower()

        if "application/json" in ct:
            data = r.json()
            sr = data.get("sr") or data.get("sample_rate") or data.get("sampling_rate")
            audio = data.get("audio") or data.get("audio_b64") or data.get("wav") or data.get("wav_b64") or data.get("data")

            if isinstance(audio, list):
                a = np.asarray(audio, dtype=np.float32)
                return a, int(sr or 24000)

            if isinstance(audio, str):
                s = audio
                if "," in s and "base64" in s[:80]:
                    s = s.split(",", 1)[1]
                b = base64.b64decode(s.encode())
                a, sr2 = self._decode_wav(b)
                return a, int(sr or sr2)

            if isinstance(audio, (bytes, bytearray)):
                a, sr2 = self._decode_wav(bytes(audio))
                return a, int(sr or sr2)

            raise RuntimeError("TTS JSON without audio")

        a, sr2 = self._decode_wav(r.content)
        return a, int(sr2)

    def _decode_wav(self, b: bytes):
        with wave.open(io.BytesIO(b), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            sw = wf.getsampwidth()
            ch = wf.getnchannels()
            raw = wf.readframes(n)

        if sw == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sw}")

        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)

        return x, int(sr)
