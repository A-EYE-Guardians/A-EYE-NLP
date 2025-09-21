import os
import sys
import time
import queue # 큐 모듈 import
import threading
import json
import pvporcupine
import sounddevice as sd
from pvporcupine import Porcupine
import struct

from dotenv import load_dotenv
import webbrowser
from pathlib import Path

# --- Import modular action handlers ---
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

load_dotenv()
print(f"PORCUPINE_ACCESS_KEY: {os.getenv('PORCUPINE_ACCESS_KEY')[:4]}...")

# --- Porcupine 설정 ---
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
WAKE_WORDS_MODEL_FILE = "C:\\Users\\ljcho\\Downloads\\A-EYE\\model\\multimodal\\stt\\헤이-류지_ko_windows_v3_0_0.ppn"
WAKE_WORD_NAME = os.path.basename(WAKE_WORDS_MODEL_FILE).split('_')[0].capitalize()

# --- 메인 함수 ---
# hotword_queue 인자를 추가합니다.
def hotword_detector_worker(hotword_queue: queue.Queue):
    """
    Porcupine 웨이크워드 감지 모듈을 실행하는 메인 함수입니다.
    """
    print("[DEBUG] hotword_detector_worker 시작됨")
    try:
        porcupine = Porcupine(
            access_key=PORCUPINE_ACCESS_KEY,
            library_path='C:\\Users\\ljcho\\Downloads\\A-EYE\\.venv\\Lib\\site-packages\\pvporcupine\\lib\\windows\\amd64\\libpv_porcupine.dll',
            sensitivities=[0.8],
            model_path='C:\\Users\\ljcho\\Downloads\\A-EYE\\model\\multimodal\\stt\\porcupine_params_ko.pv',
            keyword_paths=[WAKE_WORDS_MODEL_FILE]
        )
    except Exception as e:
        print(f"Porcupine 초기화 오류: {e}", file=sys.stderr)
        print("Access Key와 모델 파일 경로를 확인해주세요.")
        return

    try:
        #print(sd.query_devices())#을 써서
        # 마이크의 실제 인덱스(예: > 1 마이크(Realtek(R) Audio), MME (2 in, 0 out))
        # 로 selected_device_index를 교체하세요
        selected_device_index=1
        with sd.InputStream(
                samplerate=porcupine.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=porcupine.frame_length,
                device=selected_device_index
        ) as stream:
            print("[SYSTEM] 오디오 스트림이 시작되었습니다. 웨이크워드를 감지합니다. (Ctrl+C로 종료)")
            while True:
                raw_data, status = stream.read(porcupine.frame_length)

                pcm = struct.unpack('<' + 'h' * porcupine.frame_length, raw_data)

                if status:
                    print(f"오디오 스트림 경고: {status}", file=sys.stderr)

                if len(pcm) > 0:
                    result = porcupine.process(pcm)

                    if result >= 0:
                        print(f"[SYSTEM] 웨이크워드 '{WAKE_WORD_NAME}' 감지!")
                        # 핫워드가 감지되면 큐에 "hotword_detected" 같은 신호를 보냅니다.
                        hotword_queue.put("hotword_detected")
                        print(f"[DEBUG] hotword_detector_worker -> put hotword_detected, qsize={hotword_queue.qsize()}")
                else:
                    print("[SYSTEM] 오디오 데이터가 읽히지 않았습니다. 마이크 연결을 확인하세요.", file=sys.stderr)
                    break
    except KeyboardInterrupt:
        print("\n[SYSTEM] 사용자에 의해 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}", file=sys.stderr)
    finally:
        if porcupine:
            porcupine.delete()
        print("\n[SYSTEM] 프로그램이 종료되었습니다.")

# if __name__ == '__main__':
# 이 블록은 이제 `main.py`에서 호출되므로 필요하지 않습니다.
# 만약 이 스크립트를 독립적으로 테스트하고 싶다면, 아래와 같이 큐를 만들고 호출할 수 있습니다.
#    q = queue.Queue()
#    hotword_detector_worker(q)