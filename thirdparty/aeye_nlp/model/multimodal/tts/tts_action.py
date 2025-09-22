import sys
import os
import tempfile
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from openai import OpenAI

client = OpenAI()

TTS_MODEL = "tts-1"
VOICE = "onyx"

stop_flag = False


def stop_tts():
    """재생 중단"""
    global stop_flag
    stop_flag = True
    sd.stop()
    print("[TTS] 중단 요청됨.")


def handle_tts_action(text: str):
    """텍스트를 음성으로 변환 및 재생"""
    global stop_flag
    stop_flag = False

    if not text:
        print("[TTS] 합성할 텍스트 없음.")
        return

    print(f"[A-EYE]: {text}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=VOICE,
                input=text
            )
            response.stream_to_file(fp.name)
            mp3_path = fp.name

        wav_path = _convert_mp3_to_wav(mp3_path)

        if wav_path and not stop_flag:
            _play_audio_file(wav_path)

        os.remove(mp3_path)
        if wav_path:
            os.remove(wav_path)

    except Exception as e:
        print(f"[TTS ERROR] {e}", file=sys.stderr)


def _convert_mp3_to_wav(mp3_path: str) -> str:
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"[TTS 변환 오류] {e}", file=sys.stderr)
        return ""


def _play_audio_file(file_path):
    try:
        audio_data, sr = sf.read(file_path)
        sd.play(audio_data, sr)
        sd.wait()
        print("[TTS] 재생 완료.")
    except Exception as e:
        print(f"[TTS PLAY 오류] {e}", file=sys.stderr)
