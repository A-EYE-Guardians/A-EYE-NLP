from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .pipelines.registry import PIPELINE_MANAGER
from .utils.publish import UPLOAD_ROOT, ROI_DIR, append_event

app = FastAPI(title="AI Backend")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = next((p for p in [BASE_DIR / "web", Path("/app/app/web")] if p.exists()), BASE_DIR / "web")

# 정적 폴더는 /static 으로 마운트 (충돌 방지)
app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=False), name="static")

# 업로드 파일(/app/app/uploads/*) 은 /media 로 마운트
MEDIA_DIR = UPLOAD_ROOT
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR), html=False), name="media")

@app.get("/")
def root():
    return HTMLResponse('<a href="/web">Open Viewer</a>')

# 뷰어를 명시적으로 파일로 서빙 (index.html)
@app.get("/web")
@app.get("/web/index.html")
def web_index():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/pipelines/start")
def start_pipelines():
    PIPELINE_MANAGER.start()
    return {"status": "started"}

@app.post("/pipelines/stop")
def stop_pipelines():
    PIPELINE_MANAGER.stop()
    return {"status": "stopped"}

@app.get("/pipelines/status")
def status():
    return PIPELINE_MANAGER.status()


# -----------------------------------------------------------------------------
# 업로드/이벤트 수신
#  - 외부/도구(run_yoloe_pf.py 등)에서 ROI 이미지를 업로드할 수 있음
#  - 이벤트(선택 객체 메타)도 수신 받아 서버 로컬 jsonl 로 기록
# -----------------------------------------------------------------------------

@app.post("/upload")
async def upload_roi(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image/* 만 허용됩니다.")
    ROI_DIR.mkdir(parents=True, exist_ok=True)

    import time
    ts = int(time.time() * 1000)
    safe_name = file.filename or "roi.jpg"
    out_name = f"{ts}_{safe_name}"
    out_path = ROI_DIR / out_name

    data = await file.read()
    with open(out_path, "wb") as f:
        f.write(data)

    url = f"/media/roi/{out_name}"
    return {"ok": True, "url": url, "image_url": url}


@app.post("/event/selected_object")
async def event_selected_object(payload: dict = Body(...)) -> dict:
    # 서버 로컬 jsonl 로 미러 기록
    try:
        append_event(payload)
    except Exception:
        # 기록 실패는 서비스 동작을 막지 않음
        pass
    return {"ok": True}

# -----------------------------------------------------------------------------
# YOLO 선택/해제/상태
#  - 프론트에서 비디오 좌표(픽셀 또는 정규화 0~1)를 받아 YOLO 파이프라인에 전달
# -----------------------------------------------------------------------------

@app.post("/yolo/select")
async def yolo_select(x: int = Body(...), y: int = Body(...)) -> dict:
    return PIPELINE_MANAGER.yolo_select_px(x, y)


@app.post("/yolo/select_norm")
async def yolo_select_norm(x: float = Body(...), y: float = Body(...)) -> dict:
    return PIPELINE_MANAGER.yolo_select_norm(x, y)


@app.post("/yolo/clear")
async def yolo_clear() -> dict:
    return PIPELINE_MANAGER.yolo_clear()


@app.get("/yolo/status")
async def yolo_status() -> dict:
    try:
        return PIPELINE_MANAGER.status()["pipelines"]["yolo"]
    except Exception:
        return {"loaded": False, "selected": False}