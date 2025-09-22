# agent/run_graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
from tools import start_pipeline, stop_pipeline, select_norm, clear_selection, get_status, tail_events, latest_roi_base64

class S(TypedDict, total=False):
    cmd: str
    x: float
    y: float
    result: str

def node_dispatch(s: S):
    cmd = s.get("cmd","").lower()
    if cmd == "start": s["result"] = start_pipeline()
    elif cmd == "stop": s["result"] = stop_pipeline()
    elif cmd == "select":
        s["result"] = select_norm(float(s.get("x",0.5)), float(s.get("y",0.5)))
    elif cmd == "clear": s["result"] = clear_selection()
    elif cmd == "status": s["result"] = str(get_status())
    elif cmd == "events": s["result"] = str(tail_events(5))
    elif cmd == "roi":    s["result"] = latest_roi_base64() or "no roi"
    else: s["result"] = "unknown cmd"
    return s

graph = StateGraph(S)
graph.add_node("dispatch", node_dispatch)
graph.set_entry_point("dispatch")
graph.add_edge("dispatch", END)
app = graph.compile()

if __name__ == "__main__":
    # 예: 여기에 CLI/채팅루프/노트북에서 app.invoke({"cmd":"start"}) 식으로 사용
    print(app.invoke({"cmd":"status"}))
