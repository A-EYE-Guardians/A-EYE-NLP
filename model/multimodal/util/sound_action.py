# util/sound_action.py

import os
import wave
import time
import pyaudio
import openai
import faiss
import numpy as np
from pydub import AudioSegment
from pathlib import Path
from tempfile import NamedTemporaryFile

# ---------------------------
# 기본 설정
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

AUDIO_DB_PATH = Path("audio_db")
AUDIO_DB_PATH.mkdir(exist_ok=True)

INDEX_FILE = AUDIO_DB_PATH / "faiss.index"
EMB_FILE = AUDIO_DB_PATH / "embeddings.npy"
META_FILE = AUDIO_DB_PATH / "metadata.txt"


# ---------------------------
# Mini Realtime Preview 기반 AI 비서
# ---------------------------
def ai_assistant(audio_path=None, image_path=None, text=None):
    """
    오디오, 이미지, 텍스트 입력을 동시에 받아 AI 비서처럼 처리
    """
    inputs = []

    if text:
        inputs.append({"type": "input_text", "text": text})

    if audio_path:
        with open(audio_path, "rb") as f:
            inputs.append({"type": "input_audio", "data": f.read()})

    if image_path:
        with open(image_path, "rb") as f:
            inputs.append({"type": "input_image", "data": f.read()})

    response = openai.chat.completions.create(
        model="gpt-4o-mini-realtime-preview",
        messages=[{"role": "user", "content": inputs}]
    )
    return response.choices[0].message["content"]


# ---------------------------
# 로컬 오디오 DB (FAISS + Embedding) (기존 코드 그대로)
# ---------------------------
def embed_text(text):
    """텍스트 임베딩 생성"""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


def init_faiss(dimension=1536):
    """FAISS Index 초기화"""
    if INDEX_FILE.exists() and EMB_FILE.exists() and META_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        embeddings = np.load(EMB_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            metadata = f.read().splitlines()
    else:
        index = faiss.IndexFlatL2(dimension)
        embeddings = np.zeros((0, dimension), dtype="float32")
        metadata = []
    return index, embeddings, metadata


def save_faiss(index, embeddings, metadata):
    """DB 저장"""
    faiss.write_index(index, str(INDEX_FILE))
    np.save(EMB_FILE, embeddings)
    with open(META_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))


def add_audio_to_db(text, file_path):
    """오디오의 텍스트를 DB에 추가"""
    emb = embed_text(text)
    index, embeddings, metadata = init_faiss(len(emb))
    index.add(np.array([emb]))
    embeddings = np.vstack([embeddings, emb])
    metadata.append(file_path)
    save_faiss(index, embeddings, metadata)


def search_audio(query, top_k=3):
    """쿼리 텍스트로 유사한 오디오 검색"""
    query_emb = embed_text(query)
    index, embeddings, metadata = init_faiss(len(query_emb))
    if len(metadata) == 0:
        return []
    distances, ids = index.search(np.array([query_emb]), top_k)
    results = [(metadata[i], float(distances[0][j])) for j, i in enumerate(ids[0])]
    return results


# ---------------------------
# 메인 루프에서 호출할 함수
# ---------------------------
def handle_sound_action(processed_data: dict, save_dir: Path) -> str:
    """
    process_sound에서 반환된 데이터를 받아 오디오 관련 작업을 수행
    """
    command = processed_data["command"]
    action = processed_data["action"]
    file_path = processed_data.get("file_path")
    image_path = processed_data.get("image_path")

    if action == "audio_record":
        if not file_path:
            return "오디오 녹음 파일 경로를 찾을 수 없습니다."
        try:
            # 1. AI 비서에게 오디오+텍스트 통합 처리 요청
            assistant_response = ai_assistant(audio_path=file_path, text=command, image_path=image_path)

            # 2. DB에 텍스트 임베딩 저장
            add_audio_to_db(assistant_response, file_path)

            return f"AI 처리 완료: {assistant_response}, 음성이 녹음되고 데이터베이스에 추가되었습니다. (파일: {file_path})"

        except Exception as e:
            return f"오디오 녹음 처리 중 오류 발생: {e}"

    elif action == "audio_search":
        # 1. command 텍스트를 이용해 DB 검색
        results = search_audio(command)

        if results:
            # 검색 결과가 있다면
            best_match_path, similarity = results[0]
            # 여기에서 추가적으로 해당 오디오를 재생하는 기능 구현 가능
            return f"가장 유사한 오디오를 찾았습니다: '{best_match_path}', 유사도: {similarity:.2f}"
        else:
            return "유사한 오디오를 찾지 못했습니다."
'''
    elif action == "audio_describe":
        # 오디오 묘사 기능은 아직 구현되지 않았습니다.
        return "주변 소리 묘사 기능은 아직 준비 중입니다."

    else:
        return "지원하지 않는 오디오 액션입니다."
'''
# `if __name__ == '__main__':` 블록은 이 모듈을 독립적으로 실행할 때 필요하므로,
# `main.py`와 연동하려면 이 블록을 제거하거나 그대로 두셔도 상관없습니다.
# 다만, `main.py`가 이 모듈을 import할 때는 이 블록의 코드가 실행되지 않습니다.