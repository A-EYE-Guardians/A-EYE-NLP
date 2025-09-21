import os, json, glob, base64, redis
from typing import Optional
from langchain_core.tools import tool

REDIS_URL  = os.getenv("REDIS_URL", "redis://redis:6379/0") # Docker Compose에서 서비스명이 redis인 컨테이너(포트 6379, DB 0번)에 접속
CMD_STREAM = os.getenv("CMD_STREAM", "yolo_cmd") # 명령 스트림. 예) 웹/백엔드가 “이 이미지로 탐지 시작해줘” 같은 작업을 enqueue.
EVT_STREAM = os.getenv("EVT_STREAM", "yolo_events") # 이벤트 스트림. 예) 탐지 결과, 탐지 시작/중지, 오류 같은 피드백을 push.
STATUS_KEY = os.getenv("STATUS_KEY", "yolo_status") # YOLO 엔진의 현재 상태를 문자열(JSON)으로 저장
UPLOADS    = os.getenv("SHARED_UPLOADS", "/shared/uploads") # 메시지 폭주/메모리 폭증 방지

r = redis.from_url(REDIS_URL, decode_responses=True)
# 파이썬 클라이언트를 만들고, 값을 bytes가 아닌 문자열로 주고받게 설정 / JSON 직렬화/역직렬화에 편함

def _cmd(cmd: str, **kw):
    fields = {"cmd": cmd}; fields.update({k: str(v) for k,v in kw.items()})
    r.xadd(CMD_STREAM, fields, maxlen=500)

@tool("start_pipeline", return_direct=True)
def start_pipeline_tool() -> str:
    """YOLO 파이프라인을 시작합니다."""
    _cmd("start"); return "started"

@tool("stop_pipeline", return_direct=True)
def stop_pipeline_tool() -> str:
    """YOLO 파이프라인을 중지합니다."""
    _cmd("stop"); return "stopped"

@tool("select_norm", return_direct=True)
def select_norm_tool(x: float, y: float) -> str:
    """정규화 좌표(0~1)의 지점을 선택합니다. 예: x=0.5, y=0.5"""
    _cmd("select_norm", x=x, y=y); return f"select_norm({x:.3f},{y:.3f})"

@tool("clear_selection", return_direct=True)
def clear_selection_tool() -> str:
    """현재 선택을 해제합니다."""
    _cmd("clear"); return "cleared"

@tool("get_status", return_direct=True)
def get_status_tool() -> dict:
    """워커의 상태를 조회합니다."""
    h = r.hgetall(STATUS_KEY); out={}
    for k,v in h.items():
        try: out[k]=json.loads(v)
        except: out[k]=v
    return out

@tool("latest_roi_base64", return_direct=True)
def latest_roi_base64_tool() -> Optional[str]:
    """최근 ROI JPG를 base64 data URL로 반환합니다."""
    files = sorted(glob.glob(os.path.join(UPLOADS, "roi", "*.jpg")))
    if not files: return None
    with open(files[-1], "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# 툴 목록 (에이전트에서 import해서 사용)
TOOLS = [
    start_pipeline_tool, stop_pipeline_tool,
    select_norm_tool, clear_selection_tool,
    get_status_tool, latest_roi_base64_tool,
]
