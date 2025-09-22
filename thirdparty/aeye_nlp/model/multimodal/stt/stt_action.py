import os
import sys
import time
import queue
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
import soundfile as sf
from fontTools.misc.cython import returns
from numpy.distutils.system_info import language_map

# --- Configuration ---
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
MAX_SILENCE_SEC = 1
MAX_RECORD_SEC = 20

# Global objects
vad = webrtcvad.Vad(1)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_bytes(pcm_bytes: bytes, lang: str = "ko") -> str:
    """Bytes to text conversion using faster-whisper."""
    import io
    buf = io.BytesIO()
    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    sf.write(buf, int16_arr, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    segments, _ = whisper_model.transcribe(buf, beam_size=5, language=lang)
    return " ".join([seg.text for seg in segments])


def record_and_transcribe():
    """
    VAD를 사용하여 발화를 녹음하고 텍스트로 변환합니다.
    (핫워드 감지 후 호출됩니다)
    """
    q_audio = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        pcm16 = (indata * 32767).astype(np.int16).tobytes()
        q_audio.put(pcm16)

    voiced_frames = []
    silence_frames = 0
    in_speech = False
    num_padding_frames = int((MAX_SILENCE_SEC * 1000) / FRAME_MS)

    print("[STT] 발화를 녹음하고 있습니다...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback,
                        blocksize=FRAME_SIZE // 2):
        start_time = time.time()
        while True:
            if time.time() - start_time > MAX_RECORD_SEC:
                break

            if q_audio.empty():
                time.sleep(0.01)
                continue

            frame_bytes = q_audio.get()
            if len(frame_bytes) != FRAME_SIZE:
                continue

            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    in_speech = True
                    silence_frames = 0
                voiced_frames.append(frame_bytes)
            else:
                if in_speech:
                    silence_frames += 1
                    if silence_frames > num_padding_frames:
                        break

    if voiced_frames:
        utterance = b"".join(voiced_frames)
        print("[STT] 발화 감지 완료. 변환 시작...")
        return transcribe_bytes(utterance, lang="ko").strip()

    return ""

'''
if __name__ == '__main__':
    # 독립적으로 테스트하기 위한 코드
    print("STT 모듈을 테스트합니다. 말해주세요...")
    text = record_and_transcribe()
    print(f"변환된 텍스트: {text}")
'''


