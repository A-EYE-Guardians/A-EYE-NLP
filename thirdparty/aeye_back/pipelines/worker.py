# server/app/pipelines/worker.py
import os, time, json, threading
import cv2, redis
from .yolo import YoloPipeline

REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
CMD_STREAM  = os.getenv("CMD_STREAM", "yolo_cmd")
EVT_STREAM  = os.getenv("EVT_STREAM", "yolo_events")
STATUS_KEY  = os.getenv("STATUS_KEY", "yolo_status")
RTSP_URL    = os.getenv("RTSP_URL", "rtsp://mediamtx:8554/pi")

r = redis.from_url(REDIS_URL, decode_responses=True)
yolo = YoloPipeline()

_running = False
_last_id = "$"  # XREAD 처음은 최신만, 이후부터 이어받음

def set_status(**kw):
    r.hset(STATUS_KEY, mapping=kw)

def publish_event(kind, **payload):
    r.xadd(EVT_STREAM, {"kind": kind, "data": json.dumps(payload), "ts": str(int(time.time()*1000))}, maxlen=2000)

def _cmd_loop():
    global _running, _last_id
    while True:
        # block 1s, stream position from last id
        resp = r.xread({CMD_STREAM: _last_id}, block=1000, count=10)
        if not resp:
            continue
        for stream, entries in resp:
            for _id, fields in entries:
                _last_id = _id
                try:
                    cmd = fields.get("cmd")
                    if cmd == "start":
                        _running = True
                        publish_event("started")
                    elif cmd == "stop":
                        _running = False
                        publish_event("stopped")
                    elif cmd == "select_norm":
                        x = float(fields["x"]); y = float(fields["y"])
                        yolo.select_norm(x, y)
                        publish_event("select_norm", x=x, y=y)
                    elif cmd == "clear":
                        yolo.clear_selection()
                        publish_event("clear")
                    elif cmd == "status":
                        st = yolo.status() | {"running": _running}
                        publish_event("status", **st)
                except Exception as e:
                    publish_event("error", msg=str(e))
        time.sleep(0.01)

def _frame_loop():
    global _running
    cap = None
    while True:
        if _running and (cap is None or not cap.isOpened()):
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                publish_event("error", msg="rtsp_open_fail"); time.sleep(1); continue

        if not _running:
            time.sleep(0.05); continue

        ok, frame = cap.read()
        if not ok:
            publish_event("error", msg="frame_read_fail"); time.sleep(0.1); continue

        yolo.process_frame(frame)
        st = yolo.status() | {"running": True}
        set_status(**{k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else str(v) for k, v in st.items()})

        # 이벤트 발행은 yolo._maybe_publish()가 파일 저장 시 수행(기존 publish.py 사용)
        # 여기서는 주기적인 heartbeat만
        if int(time.time()) % 5 == 0:
            publish_event("heartbeat", last_shape=str(st.get("last_shape")))
        time.sleep(0.001)

if __name__ == "__main__":
    threading.Thread(target=_cmd_loop, daemon=True).start()
    _frame_loop()
