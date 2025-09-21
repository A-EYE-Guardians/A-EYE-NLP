import os, uuid, asyncio, sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

# ---- Optional: import your team's modules (A-EYE-Back/NLP/Model) ----
sys.path.extend([
    os.path.join(os.path.dirname(__file__), "thirdparty"),
    os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_back"),
    os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_model"),
    os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_nlp"),
])

PIPELINES = None
try:
    # Try to import PipelineManager from A-EYE-Back
    from aeye_back.pipelines.registry import PIPELINE_MANAGER as PIPELINES
except Exception as e:
    PIPELINES = None
    print("[LangGraph] A-EYE-Back PIPELINE_MANAGER import failed:", e)

def maybe_detect_with_team_code(roi_url: str) -> str:
    """
    Team 코드(YOLO 등)를 연결하고 싶다면 여기서 PIPELINES를 호출하세요.
    지금은 placeholder로 'cup'를 반환합니다.
    """
    try:
        if PIPELINES is not None:
            _ = PIPELINES.status()  # 예: 상태 확인
            # TODO: 실제 추론 호출로 교체
            return "cup"
    except Exception as e:
        print("[LangGraph] detect via team code failed:", e)
    return "unknown"

# ---------------------------------------------------------------------
LG_TOKEN   = os.getenv("LG_TOKEN", "LG_SECRET")       # token expected from main→/detect
MAIN_TOKEN = os.getenv("MAIN_TOKEN", "MAIN_SECRET")   # token used when calling main callbacks

def check_auth(auth: Optional[str], expected: str):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "No token")
    if auth.split(" ", 1)[1] != expected:
        raise HTTPException(403, "Bad token")

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="LangGraph API (container)")

# === (선택) 간단 헬스체크 ===
@app.get("/healthz")
def healthz():
    return {"ok": True}

# ------------------------------------------------------------
# Detect API (기존)
# ------------------------------------------------------------
class DetectReq(BaseModel):
    cid: str
    source_url: str
    params: Optional[dict] = None
    callback_base: str  # e.g., http://host.docker.internal:8000/callbacks/{cid}

class DetectAck(BaseModel):
    accepted: bool
    cid: str
    req_id: str

# (옵션) 상태/툴용 전역 상태
STATE: Dict[str, Any] = {"status": "idle", "selection": None, "last_roi_url": None}

@app.post("/detect", response_model=DetectAck)
async def detect(req: DetectReq, authorization: Optional[str] = Header(None)):
    check_auth(authorization, LG_TOKEN)
    asyncio.create_task(run_pipeline(req))
    return DetectAck(accepted=True, cid=req.cid, req_id=f"detect:{req.cid}")

async def run_pipeline(req: DetectReq):
    # 파이프라인 시작
    STATE["status"] = "running"
    try:
        # 1) 메인에 ROI 요청
        need_roi_url = f"{req.callback_base}/need_roi"
        async with httpx.AsyncClient(timeout=60.0) as client:
            need_body = {"req_id": f"R-need-roi-{uuid.uuid4()}", "hint": {"want": "latest_roi"}}
            r = await client.post(need_roi_url, json=need_body,
                                  headers={"Authorization": f"Bearer {MAIN_TOKEN}"})
            r.raise_for_status()
            roi_info = r.json()
            roi_url = roi_info.get("roi_url")
            if not roi_url:
                await send_error(req, f"main did not return roi_url: {roi_info}")
                STATE["status"] = "idle"
                return
            # 최근 ROI URL 저장(툴/디버깅용)
            STATE["last_roi_url"] = roi_url

        # 2) (옵션) 팀 코드로 실제 추론 수행
        _label_from_team = maybe_detect_with_team_code(roi_url)

        # 3) 컨펌 요청
        det_id = f"D-{uuid.uuid4()}"
        confirm_url = f"{req.callback_base}/confirm_detection"
        async with httpx.AsyncClient(timeout=60.0) as client:
            need_body = {
                "req_id": f"R-need-confirm-{uuid.uuid4()}",
                "detection_id": det_id,
                "question": "What is the detected object?",
                "options": ["cup", "mug", "bottle", "unknown"],
                "crop": {"url": roi_url, "w": 320, "h": 320},
            }
            r = await client.post(confirm_url, json=need_body,
                                  headers={"Authorization": f"Bearer {MAIN_TOKEN}"})
            r.raise_for_status()
            conf = r.json()
            label = conf.get("label") or _label_from_team or "unknown"

        # 4) 최종 결과 콜백
        final_url = f"{req.callback_base}/final"
        async with httpx.AsyncClient(timeout=60.0) as client:
            final_body = {
                "in_reply_to": f"detect:{req.cid}",
                "summary": f"1 {label} detected.",
                "detections": [{"id": det_id, "label": label, "score": 0.81}],
                "artifacts": {"annotated_url": roi_url},
            }
            await client.post(final_url, json=final_body,
                              headers={"Authorization": f"Bearer {MAIN_TOKEN}"})
    finally:
        STATE["status"] = "idle"

async def send_error(req: DetectReq, message: str):
    final_url = f"{req.callback_base}/final"
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(
            final_url,
            json={"in_reply_to": f"detect:{req.cid}", "error": message},
            headers={"Authorization": f"Bearer {MAIN_TOKEN}"},
        )

# ------------------------------------------------------------
# (옵션) Tool-style endpoints (팀 tools.py와 연동용)
# ------------------------------------------------------------
class SelectNormReq(BaseModel):
    x: float
    y: float

@app.post("/tool/start")
def tool_start():
    STATE["status"] = "running"
    return {"ok": True}

@app.post("/tool/stop")
def tool_stop():
    STATE["status"] = "idle"
    return {"ok": True}

@app.post("/tool/select_norm")
def tool_select_norm(body: SelectNormReq):
    STATE["selection"] = {"x": float(body.x), "y": float(body.y)}
    return {"ok": True, "selection": STATE["selection"]}

@app.post("/tool/clear")
def tool_clear():
    STATE["selection"] = None
    return {"ok": True}

@app.get("/tool/status")
def tool_status():
    return {"status": STATE["status"], "selection": STATE["selection"], "last_roi_url": STATE["last_roi_url"]}

@app.get("/tool/latest_roi_url")
def tool_latest_roi_url():
    return {"roi_url": STATE["last_roi_url"]}

# ------------------------------------------------------------
# 2) 랭그래프 API에 “에이전트 브릿지” 엔드포인트 추가
#    - A-EYE-Back/agent/run_graph.py 의 LangGraph 앱을 불러와 HTTP로 명령 전달
# ------------------------------------------------------------
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_back"))
    from agent.run_graph import app as BACK_AGENT_APP  # graph.compile() 결과
    print("[LangGraph] Loaded A-EYE Back agent graph successfully.")
except Exception as _e:
    BACK_AGENT_APP = None
    print("[LangGraph] Could not load A-EYE Back agent graph:", _e)

class AgentCmd(BaseModel):
    cmd: str
    args: Dict[str, Any] | None = None

@app.post("/agent/command")
async def agent_command(body: AgentCmd):
    """
    HTTP -> 팀 에이전트 그래프 브릿지.
    예)
      {"cmd":"start"}
      {"cmd":"select_norm","args":{"x":0.5,"y":0.5}}
      {"cmd":"status"}
      {"cmd":"events"}
      {"cmd":"roi"}
    """
    if BACK_AGENT_APP is None:
        raise HTTPException(500, "A-EYE Back agent not loaded in container.")
    payload: Dict[str, Any] = {"cmd": body.cmd}
    if body.args:
        payload.update(body.args)
    try:
        # run_graph.py 의 앱은 dict를 받아 dict를 반환한다고 가정
        out = BACK_AGENT_APP.invoke(payload)
        return out
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")
