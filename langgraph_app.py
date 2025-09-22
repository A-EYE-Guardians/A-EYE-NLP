\
import os, uuid, asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

# ---- Optional: import your team's modules (A-EYE-Back/NLP/Model) ----
import sys
sys.path.extend([os.path.join(os.path.dirname(__file__), "thirdparty"),
                 os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_back"),
                 os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_model"),
                 os.path.join(os.path.dirname(__file__), "thirdparty", "aeye_nlp")])

PIPELINES = None
try:
    # Try to import PipelineManager from A-EYE-Back
    from aeye_back.pipelines.registry import PIPELINE_MANAGER as PIPELINES
except Exception as e:
    PIPELINES = None
    print("[LangGraph] A-EYE-Back PIPELINE_MANAGER import failed:", e)

def maybe_detect_with_team_code(roi_url: str) -> str:
    # If your pipeline is available, you can apply detection here; 
    # for now we just return a label placeholder.
    try:
        if PIPELINES is not None:
            # Example: call yolo status or similar hooks
            _ = PIPELINES.status()
            # You can add: PIPELINES.yolo_something(...)
            return "cup"  # replace with real inference
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

class DetectReq(BaseModel):
    cid: str
    source_url: str
    params: Optional[dict] = None
    callback_base: str  # e.g., http://host.docker.internal:8000/callbacks/{cid}

class DetectAck(BaseModel):
    accepted: bool
    cid: str
    req_id: str

app = FastAPI(title="LangGraph API (container)")

@app.post("/detect", response_model=DetectAck)
async def detect(req: DetectReq, authorization: Optional[str] = Header(None)):
    check_auth(authorization, LG_TOKEN)
    # Fire-and-forget background task to simulate the pipeline
    asyncio.create_task(run_pipeline(req))
    return DetectAck(accepted=True, cid=req.cid, req_id=f"detect:{req.cid}")

async def run_pipeline(req: DetectReq):
    # 1) (simulate) ask main for ROI first
    need_roi_url = f"{req.callback_base}/need_roi"
    async with httpx.AsyncClient(timeout=60.0) as client:
        need_body = {"req_id": f"R-need-roi-{uuid.uuid4()}", "hint": {"want":"latest_roi"}}
        r = await client.post(need_roi_url, json=need_body, headers={"Authorization": f"Bearer {MAIN_TOKEN}"})
        r.raise_for_status()
        roi_info = r.json()
        roi_url  = roi_info.get("roi_url")
        if not roi_url:
            await send_error(req, f"main did not return roi_url: {roi_info}")
            return

    # 2) (simulate) detection completed → ask confirmation
    det_id = f"D-{uuid.uuid4()}"
    confirm_url = f"{req.callback_base}/confirm_detection"
    async with httpx.AsyncClient(timeout=60.0) as client:
        need_body = {
            "req_id": f"R-need-confirm-{uuid.uuid4()}",
            "detection_id": det_id,
            "question": "What is the detected object?",
            "options": ["cup","mug","bottle","unknown"],
            "crop": {"url": roi_url, "w": 320, "h": 320}
        }
        r = await client.post(confirm_url, json=need_body, headers={"Authorization": f"Bearer {MAIN_TOKEN}"})
        r.raise_for_status()
        conf = r.json()
        label = conf.get("label", "unknown")

    # 3) (simulate) finalize
    final_url = f"{req.callback_base}/final"
    async with httpx.AsyncClient(timeout=60.0) as client:
        final_body = {
            "in_reply_to": f"detect:{req.cid}",
            "summary": f"1 {label} detected.",
            "detections": [{"id": det_id, "label": label, "score": 0.81}],
            "artifacts": {"annotated_url": roi_url}
        }
        await client.post(final_url, json=final_body, headers={"Authorization": f"Bearer {MAIN_TOKEN}"})

async def send_error(req: DetectReq, message: str):
    final_url = f"{req.callback_base}/final"
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(final_url, json={"in_reply_to": f"detect:{req.cid}", "error": message},
                          headers={"Authorization": f"Bearer {MAIN_TOKEN}"})
