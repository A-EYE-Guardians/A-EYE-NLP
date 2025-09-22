from __future__ import annotations
import os, time, math
import numpy as np
import cv2

# YOLOE가 없을 수도 있으므로 안전하게 임포트
try:
    from ultralytics import YOLOE as _UL_MODEL_CLS
except Exception:
    from ultralytics import YOLO as _UL_MODEL_CLS  # fallback (weights 호환 시)
from . import registry  # 순환참조 피하려면 필요부분만 사용
from ..utils.publish import save_roi_and_get_url, append_event, build_selected_payload

class YoloPipeline:
    def __init__(self):
        self._loaded = False
        self._model = None
        self._names = {}
        self._conf = float(os.getenv("DETECTOR_CONF", "0.25"))
        self._iou  = float(os.getenv("DETECTOR_IOU", "0.50"))
        self._imgsz = int(os.getenv("DETECTOR_IMGSZ", "640"))
        self._device = os.getenv("DETECTOR_DEVICE", None)  # "cpu"/"0" 등
        self._weights = os.getenv("DETECTOR_WEIGHTS", "yoloe-11s-seg-pf.pt")

        # 선택/추적 상태
        self._selected = False
        self._prev_center = None  # (cx, cy)
        self._prev_size = None    # (w, h)
        self._pending_click = None  # (x, y) 픽셀 좌표(프레임 기준)
        self._last_send_ms = 0
        self._send_every_ms = int(os.getenv("YOLO_SEND_EVERY_MS", "300"))
        self._source_label = os.getenv("YOLO_SOURCE_LABEL", "rtsp:consumer")

        # 마지막 프레임 크기 (정규화 선택 좌표 변환용)
        self._last_shape = None  # (h, w)

    def _load(self):
        if self._loaded:
            return
        self._model = _UL_MODEL_CLS(self._weights)
        # names 확보
        try:
            self._names = getattr(self._model, "names", {}) or {}
        except Exception:
            self._names = {}
        self._loaded = True

    # 외부에서 선택 지정(픽셀 좌표)
    def select(self, x: int, y: int):
        self._pending_click = (int(x), int(y))

    # 외부에서 선택 지정(정규화 좌표 0~1)
    def select_norm(self, x_norm: float, y_norm: float):
        if self._last_shape is None:
            return
        h, w = self._last_shape
        x = int(max(0, min(1.0, x_norm)) * w)
        y = int(max(0, min(1.0, y_norm)) * h)
        self.select(x, y)

    def clear_selection(self):
        self._selected = False
        self._prev_center = None
        self._prev_size = None
        self._pending_click = None

    def _pick_by_point(self, xyxy: np.ndarray, click_xy):
        if xyxy.size == 0: return -1
        x, y = click_xy
        inside_idx, inside_area = [], []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                inside_idx.append(i); inside_area.append((x2 - x1) * (y2 - y1))
        if inside_idx:
            return inside_idx[int(np.argmin(inside_area))]
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2]); cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        return int(np.argmin(d2))

    def _reassign_by_prev(self, xyxy: np.ndarray):
        if xyxy.size == 0 or self._prev_center is None: return -1
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2]); cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        d = np.sqrt((cx - self._prev_center[0]) ** 2 + (cy - self._prev_center[1]) ** 2)
        idx = int(np.argmin(d))
        diag = math.hypot(*(self._prev_size or (80.0, 80.0)))
        return idx if d[idx] <= diag * 1.2 else -1

    def _maybe_publish(self, frame, xyxy, cls_id, conf):
        now = int(time.time() * 1000)
        if now - self._last_send_ms < self._send_every_ms:
            return
        self._last_send_ms = now

        label_name = self._names.get(int(cls_id), str(int(cls_id)))
        url = save_roi_and_get_url(frame, tuple(map(int, xyxy)), label_name)
        payload = build_selected_payload(frame, xyxy, label_name, float(conf), self._source_label, url)
        append_event(payload)

    def process_frame(self, bgr):
        if not self._loaded:
            self._load()
        if bgr is None:
            return None

        h, w = bgr.shape[:2]
        self._last_shape = (h, w)

        results = self._model.predict(
            source=bgr, imgsz=self._imgsz, conf=self._conf, iou=self._iou,
            agnostic_nms=False, max_det=30, device=self._device, stream=False, verbose=False
        )
        res = results[0]
        boxes = getattr(res, "boxes", None)
        names = getattr(res, "names", None) or self._names
        if names and not self._names:
            self._names = names

        selected_idx = -1
        xyxy = np.empty((0, 4), dtype=np.float32)
        cls = conf = None

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            # 새 클릭 들어왔으면 가까운 박스 선택
            if self._pending_click is not None:
                selected_idx = self._pick_by_point(xyxy, self._pending_click)
                self._selected = selected_idx >= 0
                self._pending_click = None

                if self._selected:
                    x1, y1, x2, y2 = xyxy[selected_idx]
                    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
                    self._prev_center = (float(cx), float(cy))
                    self._prev_size   = (float(x2-x1), float(y2-y1))
                    self._maybe_publish(bgr, (x1,y1,x2,y2), cls[selected_idx], conf[selected_idx])

            # 추적 유지 시 재연결 + 주기 전송
            if self._selected and self._prev_center is not None:
                re_idx = self._reassign_by_prev(xyxy)
                if re_idx >= 0:
                    selected_idx = re_idx
                    x1, y1, x2, y2 = xyxy[selected_idx]
                    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
                    self._prev_center = (float(cx), float(cy))
                    self._prev_size   = (float(x2-x1), float(y2-y1))
                    self._maybe_publish(bgr, (x1,y1,x2,y2), cls[selected_idx], conf[selected_idx])
                else:
                    self.clear_selection()

        return None  # 필요시 시각화 프레임/메타 반환

    def status(self):
        return {
            "loaded": self._loaded,
            "selected": self._selected,
            "last_shape": self._last_shape,
            "send_every_ms": self._send_every_ms,
        }
