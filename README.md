# 이 레포지토리는 A-EYE 프로젝트 Langgraph를 이용한 NLP 처리 레포입니다.
=======

# 🎤 A-EYE 멀티모달 비서 시스템

본 프로젝트는 **음성 인식(STT)**, **자연어 처리(LLM)**, **이미지/문서 처리**, **음성 합성(TTS)**, 그리고 **핫워드 감지**를 통합한 **멀티모달 AI 비서** 시스템입니다.  
아래는 팀원들이 전체 구조와 원리를 이해할 수 있도록 상세히 작성된 기술 설명 및 코드 리뷰 성격의 README입니다.

---

## 🏗️ 전체 파이프라인 개요

1. **웨이크워드(Hotword) 감지**
   - 사용자가 "헤이-류지"라고 발화하면 시스템이 STT 모드로 전환됩니다.
   - 구현: `webrtcvad` + PyAudio → 실시간 오디오 스트림 분석 → 큐(`queue.Queue`)에 이벤트 전달.

2. **STT (Speech-to-Text)**
   - 발화된 음성을 텍스트로 변환.
   - 지원 엔진: `vosk`, `whisper(faster-whisper)`
   - 변환된 텍스트는 명령 큐(`command_queue`)로 들어감.

3. **명령어 처리 & Reflection**
   - LLM(OpenAI API 기반)을 통해 사용자의 발화를 **의도(action)** + **매개변수(params)** 구조로 파싱.
   - Reflection 기법: 이전 결과/실패를 참고하여 LLM이 더 나은 action을 스스로 수정 가능.

4. **Action 실행**
   - `process_command`에서 action 유형에 맞는 `action` 모듈 호출.
   - 예시:
     - `reply`: 일반 대화
     - `object_info`: 특정 객체에 대한 정보
     - `text_recognition`: 이미지 속 글자 인식
     - `scan_code`: 바코드나 QR 코드 읽기
     - `control_hw`: 하드웨어 제어
     - `audio_search`: 소리 기반 검색
     - `audio_record`: 녹음
     - `audio_describe`: 소리 해설
     - `navigate_image`: 방향 표시
     - `answer_by_document`: 문서 기반 답변
     - `document_summary`: 문서 요약
     - `transcribe_audio`: 받아쓰기
     - `qa_generation`: 문제 출제
     - `highlight`: 중요 표시할만한 내용 추출
     - `compare_documents`: 비교(유사도)
     - `translation`: 번역
     - `timeline_generation`: 시간대별 정리
     - `code_extraction`: 코딩된(프로그래밍 언어로 이뤄진) 부분 추출
     - `search_papers`: 전문 자료(논문 등) 검색 및 정리
   - 실행 결과는 다시 `state`에 저장.

5. **TTS (Text-to-Speech)**
   - 응답 텍스트를 음성으로 변환 후 출력.
   - 엔진: `pyttsx3`, 또는 `gTTS`(선택 가능).

6. **세션 관리**
   - `is_active_session`, `last_interaction_time` 변수를 통해 세션 지속 여부 결정.
   - 종료 조건:
     - 종료 발화("바이 류지" 등) 감지
     - TTS 출력 후 일정 시간(예: 20초) 입력 없음 → 자동 종료

---

## 🧩 사용된 기술 스택

### 🔊 오디오 처리
- **PyAudio**: 마이크 입력 스트림 처리
- **webrtcvad**: Google의 Voice Activity Detection → 음성 여부 감지
- **queue.Queue**: 멀티스레드 환경에서 이벤트 전달

### 🗣️ 음성 인식 (STT)
- **Vosk**: 오픈소스 STT 엔진 (한국어 모델 지원)
- **Whisper**: OpenAI 모델, 정확도 높음 (속도/리소스 trade-off)

### 🤖 자연어 처리
- **OpenAI API (GPT 계열)**:
  - 사용자 입력 → Action/Params 구조 추출
  - Reflection 기법으로 실패 시 재시도/수정

### 🖼️ 멀티모달 처리
- **이미지 처리**: OpenCV, Pillow, PyTorch 등 활용
- **문서 처리**: PDF/Word/웹 문서 전처리기 구현

### 🔉 음성 합성 (TTS)
- **pyttsx3**: 로컬 TTS (속도 빠름, 인터넷 불필요)
- **gTTS**: Google TTS (더 자연스러움)

### ⚙️ 실행 환경
- **Python 3.10+**
- **Docker**: 팀원 간 환경 차이를 최소화하기 위해 도입
  - `Dockerfile`에 requirements 고정
  - GPU 활용 시 `nvidia-docker` 권장

---

## 📜 코드 동작 원리 (Step-by-Step)

### 1) 전역 상태 및 큐
```python
hotword_detected_queue = queue.Queue()
command_queue = queue.Queue()
running = True
last_interaction_time = time.time()
is_active_session = False
````

* 큐를 통해 **스레드 간 안전한 이벤트 전달** 수행.
* `is_active_session`은 현재 STT 모드 여부를 나타냄.

---

### 2) 메인 루프

```python
while running:
    if not hotword_detected_queue.empty():
        command_queue.put('hotword_detected')

    command = command_queue.get(timeout=0.1)

    if command == 'hotword_detected':
        is_active_session = True
        user_text = record_and_transcribe()
        process_with_reflection(user_text, state)
```

* 핫워드 감지 시 STT 모드 진입.
* 입력을 STT 또는 텍스트로 받아 LLM 처리기로 보냄.

---

### 3) Reflection 처리

```python
state = process_with_reflection(user_input, state)
```

* LLM이 action을 추론.
* 실패 시 이전 결과를 피드백 → **자기수정(Self-Reflection)** 가능.

---

### 4) TTS 출력

```python
handle_tts_action(result_text)
last_interaction_time = time.time()
```

* 응답을 음성으로 출력.
* TTS 완료 시점을 기준으로 타이머 리셋 → UX 개선.

---

### 5) 세션 종료 조건

```python
if state.get('pending_shutdown') or (time.time() - last_interaction_time) > SESSION_TIMEOUT:
    is_active_session = False
```

* 종료 발화 감지 → 즉시 종료
* 또는 타임아웃(예: 20초) → 자동 종료

---

## 🚀 실행 방법

### 1) 로컬 실행

```bash
python main.py
```

### 2) Docker 실행

```bash
docker build -t a-eye .
docker run --rm -it --device /dev/snd --name aeye a-eye
```

* `--device /dev/snd` 옵션 필요 (마이크 접근 허용)

---

## ✅ 핵심 포인트

1. **핫워드 감지 → STT → LLM → Action → TTS** 순서로 동작.
2. 모든 모듈은 **스레드 + 큐 기반 이벤트 처리** 구조로 묶임.
3. 세션 관리 로직이 UX 품질에 직접적인 영향을 줌.
4. Docker 환경을 쓰면 팀원 간 버전/패키지 충돌을 막을 수 있음.

---
