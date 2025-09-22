# processing_sound.py
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import tempfile
import os
from pathlib import Path

SAMPLE_AUDIO_PATH = ("./loader/sample.wav")  # fallback 오디오 파일

def record_audio(duration: int = 5, samplerate: int = 16000) -> np.ndarray:
    """마이크에서 일정 시간 동안 오디오 녹음"""
    print(f"[SYSTEM] {duration}초 동안 오디오 녹음 시작...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("[SYSTEM] 녹음 완료")
    return audio.flatten()

def load_audio(file_path: str, samplerate: int = 16000) -> torch.Tensor:
    """파일에서 오디오 로드 후 Tensor 반환"""
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != samplerate:
            transform = torchaudio.transforms.Resample(sr, samplerate)
            waveform = transform(waveform)
        return waveform
    except Exception as e:
        print(f"[경고] 오디오 로드 실패({e}), 샘플 오디오 사용")
        waveform, sr = torchaudio.load(SAMPLE_AUDIO_PATH)
        return waveform

def audio_to_mel(waveform: torch.Tensor, samplerate: int = 16000, n_mels: int = 64) -> torch.Tensor:
    """waveform → Mel spectrogram 변환"""
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=samplerate,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels
    )
    mel = mel_spectrogram(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel_db

def process_sound(action: dict, command: str) -> dict:
    """
    JSON + command -> 오디오 데이터와 함께 임시 파일 경로를 반환
    """
    act = action.get("action")
    params = action.get("params", {})
    file_path = None # file_path 변수 초기화

    if act == "audio_record":
        duration = params.get("duration", 5)
        audio = record_audio(duration=duration)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
            torchaudio.save(tmpf.name, torch.tensor(audio).unsqueeze(0), 16000)
            file_path = tmpf.name # 임시 파일 경로를 저장
        waveform = load_audio(file_path)
    else:
        file_path = params.get("file_path", SAMPLE_AUDIO_PATH)
        waveform = load_audio(file_path)

    mel_tensor = audio_to_mel(waveform)

    return {
        "command": command,
        "action": act,
        "waveform": waveform,
        "mel_tensor": mel_tensor,
        "params": params,
        "file_path": file_path  # 임시 파일 경로를 반환 딕셔너리에 추가
    }