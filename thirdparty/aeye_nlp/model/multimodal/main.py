# main.py (통합판)
import json
import os
import re
import queue
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from stt.hotword_detector import hotword_detector_worker
from stt.stt_action import record_and_transcribe
from tts.tts_action import handle_tts_action
from util.search_utils import google_search

from loader.processing_image import process_image
from util.image_action import handle_image_action
from loader.processing_sound import process_sound
from util.sound_action import handle_sound_action
from util.document_action import handle_document_action

# (옵션) reflexion pipeline 모듈이 있으면 불러와서 문서 전용 처리에 쓸 수 있음
try:
    from reflextion_graph import run_reflexion_pipeline
    HAS_REFLEXION_GRAPH = True
except Exception:
    HAS_REFLEXION_GRAPH = False

# --------------- 환경 변수 및 클라이언트 ---------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if client is None:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")

# --------------- 전역 큐/상태 ---------------
hotword_detected_queue = queue.Queue()
command_queue = queue.Queue()  # 텍스트 입력 등 일반 명령
running = True
MULTIMODAL_FILE_PATH = Path("./saved_result")
conversation_history = []
SESSION_TIMEOUT = 30
last_interaction_time = time.time()
is_active_session = False

# 액션 분류 목록 (너가 정의한 것 재사용)
IMAGE_ACTIONS = ["object_info", "text_recognition", "scan_code", "control_hw", "navigate_image"]
DOCUMENT_ACTIONS = ["document_summary", "code_extraction", "timeline_generation", "translation",
                    "qa_generation", "highlight", "compare_documents", "answer_by_document", "search_papers"]
SOUND_ACTIONS = ["audio_search", "audio_record", "audio_describe", "transcribe_audio"]

# ---------------- 동기 관리 ----------------

tts_playing = False

def tts_thread_func(text):
    global tts_playing
    tts_playing = True
    handle_tts_action(text)  # 기존 블로킹 함수
    tts_playing = False

# ---------------- 유틸리티 ----------------
def push_history(role: str, content: str):
    global conversation_history, last_interaction_time
    conversation_history.append({"role": role, "content": content, "time": time.time()})
    last_interaction_time = time.time()

def safe_json_load(s: str):
    """GPT가 보내는 JSON-ish 문자열을 안전하게 파싱하는 best-effort helper."""
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        # code fence 제거
        lines = s.splitlines()
        if len(lines) >= 3:
            s = "\n".join(lines[1:-1])
    # 흔한 single quote 문제 대처
    if s.startswith("{'") or "'action'" in s:
        s = s.replace("'", '"')
    # try to extract JSON object substring
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        try:
            # 마지막 수단으로 eval (제한된 환경)
            obj = eval(s, {"__builtins__": {}}, {})
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

# ----------------- LLM 기반 action 추론 -----------------
def infer_action(command: str) -> dict:
    """
    기존 infer_action을 거의 그대로 사용.
    사용자 발화 -> action JSON을 반환.
    실패시 {'action':'reply','params':{'text':<원문>}}
    """
    system_prompt = (
        "너는 AR 안경의 AI 비서야. "
        "오입력, 오인식 등을 고려한 user의 발화를 분석해서 적절한 JSON action으로 변환해. "
        "이거 등과 같은 추상적 지칭이 포함돼 외부 하드웨어의 입력이 필요해 보이는 명령은 "
        "사운드나 이미지 관련 action일 가능성이 높다고 가정해."
        "각 action: " 
        "object_info, 특정 객체에 대한 정보. " 
        "text_recognition, 이미지 속 글자 인식. " 
        "scan_code, 바코드나 QR 코드 읽기. " 
        "control_hw, 하드웨어 제어. " 
        "audio_search, 소리 기반 검색. " 
        "audio_record, 녹음. " 
        "audio_describe, 소리 해설. " 
        "navigate_image, 방향 표시. " 
        "answer_by_document, 문서 기반 답변. " 
        "document_summary, 요약. " 
        "transcribe_audio, stt. " 
        "qa_generation, 문제 출제. " 
        "highlight, 중요 표시할 내용. " 
        "compare_documents, 비교. " 
        "translation, 번역. " 
        "timeline_generation, 시간대 답변. " 
        "code_extraction, 코딩한 부분 추출. " 
        "search_papers, 검색. " 
        "reply, 위 action이 아닌 모든 일반 대화. "
        "반환은 JSON 하나만, 형식은 {\"action\":\"...\",\"params\":{...}}."
    )
    if client is None:
        return {"action":"reply","params":{"text":"OpenAI 키가 설정되어 있지 않습니다."}}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":command}
            ],
            temperature=0,
            max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        parsed = safe_json_load(raw)
        if parsed and isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print(f"[GPT ERROR infer_action] {e}", file=sys.stderr)
    return {"action":"reply","params":{"text":command}}

# ----------------- Reflection 단계: 초안 -> 개선안(액션) -----------------
def improve_action_via_reflection(user_command: str, draft_action: dict, memory_context: str = None) -> dict:
    """
    1) draft_action(JSON)과 원문 user_command를 LLM에 주고
    2) 개선된 action JSON을 반환하도록 요청(이게 '반성'의 핵심)
    반환값: dict action (fallback: draft_action)
    """
    system = (
        "당신은 자가반성(Reflexion) 모듈입니다. 사용자의 명령과 초기 액션 초안을 보고, "
        "실제 실행에 더 적절한 형태의 action JSON을 생성하세요. "
        "반드시 JSON 하나만 반환하세요. (예: {\"action\":\"document_summary\",\"params\":{\"query\":\"튜링 테스트\"}} )"
    )
    user_msg = f"원문: {user_command}\n\n초안액션: {json.dumps(draft_action, ensure_ascii=False)}"
    if memory_context:
        user_msg += f"\n\n기억: {memory_context}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system}, {"role":"user","content":user_msg}],
            temperature=0.0,
            max_tokens=300
        )
        raw = resp.choices[0].message.content
        parsed = safe_json_load(raw)
        if parsed and isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print(f"[GPT ERROR improve_action] {e}", file=sys.stderr)
    # fallback
    return draft_action

# ----------------- Action 실행기 -----------------
def execute_action(action: dict, user_command: str, state: dict):
    """
    action(dict) 실행: 여러 핸들러들 호출.
    반환: (result_text (str) or structured object)
    """
    if not isinstance(action, dict):
        return "[ERROR] 액션 포맷 불일치"

    act_name = action.get("action", "reply")
    params = action.get("params", {}) or {}

    try:
        if act_name == "reply":
            # 간단한 대화 (기존 conversation memory 사용)
            # chat_with_memory는 긴 텍스트를 반환할 수 있음
            res = chat_with_memory(user_command)
            return res

        if act_name in IMAGE_ACTIONS:
            # 이미지 action: action(dict) -> process_image -> handle_image_action
            processed = process_image(action, user_command)
            # save tensor to state for possible follow-up
            state["last_image_tensor"] = processed.get("image_tensor")
            res = handle_image_action(processed, MULTIMODAL_FILE_PATH)
            return res

        if act_name in SOUND_ACTIONS:
            processed = process_sound(action, user_command)
            res = handle_sound_action(processed, MULTIMODAL_FILE_PATH)
            return res

        if act_name in DOCUMENT_ACTIONS:
            # 문서 관련: run_reflexion_pipeline 우선(있다면), 아니면 document_action handler
            if HAS_REFLEXION_GRAPH and params.get("use_reflexion", True):
                final_answer, traj = run_reflexion_pipeline(user_command)
                return final_answer
            else:
                # 💡 수정된 부분: handle_document_action의 인자 포맷에 맞게 데이터 추출
                # 'query' 파라미터가 없으면 user_command를 기본값으로 사용
                query = params.get("query", user_command)
                sources = params.get("sources")
                top_k = params.get("top_k", 3)

                # handle_document_action에 올바른 인자 전달
                return handle_document_action(query, sources, top_k)

            # 검색 액션 (임의: action 'search' 혹은 'search_papers')
        if act_name in ("search", "search_papers"):
            q = params.get("query") or user_command
            return google_search(q, num_results=params.get("num_results", 5), sleep_sec=1)

        # fallback unknown
        return f"[ERROR] 미정의 액션: {act_name}"

    except Exception as e:
        print(f"[EXECUTION ERROR] {e}", file=sys.stderr)
        return f"[ERROR] 실행 중 예외: {e}"

# ----------------- 보조: 기존 chat_with_memory (re-usable) -----------------
def chat_with_memory(user_text: str):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_text})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"너는 전문 문서 분석 및 웹 검색용 AI야."}] + conversation_history,
            temperature=0.7
        )
        reply = resp.choices[0].message.content.strip()
        conversation_history.append({"role":"assistant","content":reply})
        # TTS 최적화: 길면 요약
        if len(reply) > 200:
            sresp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"긴 답변을 200자 내외로 요약해줘."},{"role":"user","content":reply}],
                temperature=0
            )
            return sresp.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print(f"[GPT ERROR chat_with_memory] {e}", file=sys.stderr)
        return "[ERROR] LLM 오류로 응답 불가"

# ----------------- 전체 파이프라인: 반성 전/후 포함 -----------------
def process_with_reflection(user_input: str, state: dict):
    """
    전체 플로우:
    1) run_reflexion_pipeline (actor -> search -> fetch -> evaluate -> reflect -> revision) 시도
       -> final_answer, traj (tool_outputs 포함)
    2) infer_action (초안)
    3) improve_action_via_reflection(초안 + evidence) -> 개선된 액션
    4) 결정: 개선된 액션이 reply이면 final_answer 우선 출력. 아니면 액션 실행.
    """
    global last_interaction_time
    final_answer = None
    traj = None

    # 1) Reflexion pipeline 시도 (가능하면)
    if HAS_REFLEXION_GRAPH:
        try:
            # 💡 수정된 부분: run_reflexion_pipeline이 반환하는 값을 정확히 받습니다.
            # 만약 pipeline 내에서 에러가 발생하면, 아래 except 블록에서 잡아냅니다.
            final_answer, traj = run_reflexion_pipeline(user_input)
        except Exception as e:
            print(f"[REFLEXION ERROR] {e}", file=sys.stderr)
            final_answer, traj = None, None

    # 2) Action 초안
    draft = infer_action(user_input)

    # 3) evidence (tool outputs) 형태로 추출해서 improve 함수에 전달
    evidence = None
    try:
        if traj:
            tool_list = getattr(traj, "tool_outputs", None)
            if tool_list is None and isinstance(traj, dict):
                tool_list = traj.get("tool_outputs")
            if tool_list:
                evidence = "\n\n".join(tool_list)[:3500]
    except Exception:
        evidence = None

    # 4) 개선된 액션 생성 (draft + evidence)
    improved = improve_action_via_reflection(user_input, draft, memory_context=evidence)

    # 안전장치
    if not isinstance(improved, dict) or "action" not in improved:
        improved = draft

    # 5) Decision: reply이면 final_answer 먼저 사용(있다면)
    result = None
    if improved.get("action") == "reply":
        if final_answer:
            result = final_answer
        else:
            result = chat_with_memory(user_input)
    else:
        if evidence:
            improved.setdefault("params", {})["evidence"] = evidence
        result = execute_action(improved, user_input, state)
        if (not result or (isinstance(result, str) and not result.strip())) and final_answer:
            result = final_answer

    # 6) 간단 평가/반성(선택적)
    try:
        eval_prompt = (
            f"사용자 명령: {user_input}\n\n실행된 액션: {json.dumps(improved, ensure_ascii=False)}\n\n"
            f"실행결과(요약): {str(result)[:2000]}\n\n"
            "이 실행의 문제점(있다면)과 개선할 점을 1-2문장으로 적어라."
        )
        eval_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "당신은 평가자입니다."},
                      {"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        evaluation = eval_resp.choices[0].message.content.strip()
    except Exception:
        evaluation = None

    # 7) TTS 및 상태 업데이트
    result_text = ""
    if isinstance(result, (dict, list)):
        try:
            result_text = json.dumps(result, ensure_ascii=False)[:1500]
        except Exception:
            result_text = str(result)[:1500]
    else:
        result_text = str(result) if result is not None else "[EMPTY]"

    handle_tts_action(result_text)
    # TTS 끝나고 last_interaction_time 갱신
    last_interaction_time = time.time()
    result_text = str(result) if result is not None else "[EMPTY]"
    threading.Thread(target=tts_thread_func, args=(result_text,), daemon=True).start()
    push_history("assistant", result_text)
    state["last_command"] = user_input
    state["last_action"] = improved
    state["last_result"] = result_text
    return state


# ----------------- 키보드 리스너 (stdin) -----------------
def keyboard_listener_worker():
    while running:
        try:
            line = sys.stdin.buffer.readline().decode("utf-8", errors="ignore")
            if not line:
                time.sleep(0.1)
                continue
            user_input = line.strip()
            if not user_input:
                continue
            # 특수 토큰 처리
            if user_input.lower() == "as":
                command_queue.put("as_command")
            elif user_input.lower() == "sa":
                command_queue.put("sa_command")
            else:
                command_queue.put(user_input)
        except Exception as e:
            print(f"[KEYBOARD LISTENER ERROR] {e}", file=sys.stderr)
            time.sleep(0.5)

# ----------------- 메인 루프 -----------------
def main():
    global running, last_interaction_time, is_active_session

    hotword_thread = threading.Thread(
        target=hotword_detector_worker,
        args=(hotword_detected_queue,),
        daemon=True)
    hotword_thread.start()

    keyboard_thread = threading.Thread(target=keyboard_listener_worker, daemon=True)
    keyboard_thread.start()

    time.sleep(1)
    print("as = txt, sa = stt: ", end='', flush=True)

    # 상태 dict 초기화
    state = {
        "action": "",
        "params": {},
        "command_text": "",
        "result": "",
        "last_image_tensor": None,
        "last_command": "",
        "last_result": ""
    }

    while running:
        try:
            # 핫워드 감지 결과 우선 처리
            if not hotword_detected_queue.empty():
                hotword_detected_queue.get_nowait()
                print("[DEBUG] 웨이크워드 감지 신호 받기 직전")
                command_queue.put('hotword_detected')
                print("[DEBUG] 웨이크워드 감지 신호 put 완료")
                hotword_detected_queue.task_done()

            # 큐에서 명령 대기 (is_active_session에 따라 timeout 조절)
            timeout = 0.1 #if is_active_session else None
            try:
                print("[DEBUG] command_queue.get 대기중")
                command = command_queue.get(timeout=timeout)
            except queue.Empty:
                command = None
            print("[SYSTEM] 웨이크워드 감지: ", {command})

            # 핫워드 처리: STT → 처리
            if command is None:
                continue  # 큐 비어있으면 루프 처음으로

            last_interaction_time = time.time()

            # 사용자가 입력했을 때만 last_interaction_time 갱신
            if command is not None and command not in ("hotword_detected", "as_command", "sa_command"):
                last_interaction_time = time.time()  # 사용자가 말하거나 입력했을 때만

            if command == 'hotword_detected':
                is_active_session = True
                print("[SYSTEM] STT 모드 진입")
                user_text = record_and_transcribe()
                last_interaction_time = time.time()
                if user_text:
                    print(f"[SYSTEM] 감지된 음성 명령어: '{user_text}'")

                    cooldown = ["땡큐 류지", "바이 류지", "탱규류지", "땐큐류지", "바이류지", "바이유지", "탱규 류지", "땐큐 류지", "바이 류지", "바이 유지", "탱규루지", "땐큐루지", "바이루지", "바이루지"]
                    if user_text != cooldown :
                        state = process_with_reflection(user_text, state)
                    else:
                        # 종료 발화 감지 → TTS 끝난 후 pending_shutdown=True
                        state['pending_shutdown'] = True
                        # TTS로 종료 안내
                        handle_tts_action("세션을 종료합니다.")
                        # 마지막 상호작용 시간 갱신: TTS 종료 시점
                        last_interaction_time = time.time()
                        print("[SYSTEM] 종료 입감 → 핫워드 대기")
                        is_active_session = False
                else:
                    print("[SYSTEM] 음성 입력 없음 → 핫워드 대기")
                    is_active_session = False

                    # 키보드/텍스트 입력 처리

                    try:
                        command = command_queue.get(timeout=0.1)
                    except queue.Empty:
                        command = None
                    if command:
                        if command in ("as_command", "sa_command"):
                            is_active_session = True
                            print(f"[SYSTEM] {command} 모드 진입")
                        else:
                            state = process_with_reflection(command, state)
                        command_queue.task_done()
            # task done safe
            try:
                command_queue.task_done()
            except Exception:
                pass

        except queue.Empty:
            # 세션 타임아웃 체크
            if is_active_session and (
                    state.get('pending_shutdown') or (time.time() - last_interaction_time) > SESSION_TIMEOUT):
                print("[SYSTEM] 세션 종료 → 핫워드 대기")
                is_active_session = False
                state['pending_shutdown'] = False
        except KeyboardInterrupt:
            print("[SYSTEM] KeyboardInterrupt - 종료 중...")
            running = False
            break
        except Exception as e:
            print(f"[SYSTEM ERROR] 루프 예외: {e}", file=sys.stderr)
            # 세션 리셋
            is_active_session = False

    # Cleanup
    hotword_thread.join()
    keyboard_thread.join()

if __name__ == "__main__":
    main()
