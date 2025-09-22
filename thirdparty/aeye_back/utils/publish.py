from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2, time, json, os

# 업로드/미디어 루트: /app/app/uploads
UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploads"
ROI_DIR = UPLOAD_ROOT / "roi"
EVENT_LOG = UPLOAD_ROOT / "events.jsonl"
ROI_DIR.mkdir(parents=True, exist_ok=True)
EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)

def _safe_crop_jpeg(frame_bgr, xyxy: Tuple[int,int,int,int], quality: int = 90) -> bytes:
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: {xyxy}")
    roi = frame_bgr[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode ROI JPEG")
    return buf.tobytes()

def save_roi_and_get_url(frame_bgr, xyxy: Tuple[int,int,int,int], label: str) -> str:
    ts_ms = int(time.time() * 1000)
    fname = f"{ts_ms}_{label}.jpg"
    jpeg = _safe_crop_jpeg(frame_bgr, xyxy)
    (ROI_DIR).mkdir(parents=True, exist_ok=True)
    with open(ROI_DIR / fname, "wb") as f:
        f.write(jpeg)
    # /media 마운트가 main.py에 추가됩니다
    return f"/media/roi/{fname}"

def append_event(payload: Dict[str, Any]) -> None:
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def build_selected_payload(frame_bgr, xyxy, label: str, conf: float, source: str, image_url: str) -> Dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    return {
        "event": "selected_object",
        "ts_ms": ts_ms,
        "source": source,
        "frame_size": {"w": w, "h": h},
        "label": label,
        "conf": float(conf),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1},
        "caption": f"{label}로 보입니다.",
        "image_url": image_url,
    }