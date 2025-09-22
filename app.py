from fastapi import APIRouter
from langgraph_app import app as base_app
from agent.tools import (
    get_status_tool, latest_roi_base64_tool,
    start_pipeline_tool, stop_pipeline_tool,
    select_norm_tool, clear_selection_tool,
)

def _run_tool(t, payload=""):
    # LangChain Tool이면 .invoke 사용, 아니면 함수처럼 호출
    if hasattr(t, "invoke"):
        return t.invoke(payload)
    return t(payload)

app = base_app
router = APIRouter(prefix="/agent", tags=["agent-tools"])

@router.get("/status")
def status():
    # ❗️기존: return get_status_tool()
    return _run_tool(get_status_tool, "")

@router.get("/roi")
def roi():
    # ❗️기존: latest_roi_base64_tool()
    return {"data_url": _run_tool(latest_roi_base64_tool, "")}

@router.post("/start")
def start():
    return {"ok": True, "result": _run_tool(start_pipeline_tool, "")}

@router.post("/stop")
def stop():
    return {"ok": True, "result": _run_tool(stop_pipeline_tool, "")}

@router.post("/select")
def select(x: float, y: float):
    # 좌표를 payload로 넘기기
    return {"ok": True, "result": _run_tool(select_norm_tool, {"x": x, "y": y})}

@router.post("/clear")
def clear():
    return {"ok": True, "result": _run_tool(clear_selection_tool, "")}

app.include_router(router)
