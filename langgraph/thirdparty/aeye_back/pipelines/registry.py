from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional, Tuple

from ..consumers.rtsp_consumer import RTSPConsumer
from .yolo import YoloPipeline
from .stt import STTPipeline
from .eye import EyePipeline

RTSP_URL = os.getenv('RTSP_URL', 'rtsp://localhost:8554/pi')


class PipelineManager:
    def __init__(self) -> None:
        self.consumer = RTSPConsumer(RTSP_URL)
        self.yolo = YoloPipeline()
        self.stt = STTPipeline()
        self.eye = EyePipeline()

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def _get_from_consumer(self, timeout: float = 0.5) -> Optional[Tuple[str, Any]]:
        try:
            return self.consumer.get(timeout=timeout)
        except TypeError:
            # timeout 인자 미지원인 구현 대비
            try:
                return self.consumer.get()
            except Exception:
                time.sleep(timeout)
                return None        
    
    def _loop(self) -> None:
        self._running = True
        while self._running:
            item = self._get_from_consumer(timeout=0.5)
            if not item:
                continue

            kind, data = item  # ('video', frame_bgr) or ('audio', (pcm16, sr))

            if kind == "video":
                frame_bgr = data
                # 프레임 단위 파이프라인 호출
                try:
                    self.yolo.process_frame(frame_bgr)
                except Exception:
                    # 파이프라인 오류 격리
                    pass
                try:
                    self.eye.process_frame(frame_bgr)
                except Exception:
                    pass

            elif kind == "audio":
                try:
                    pcm, sr = data
                    self.stt.process_audio(pcm, sr)
                except Exception:
                    pass

    def start(self) -> dict:
        if self._running:
            return {"ok": True, "running": True, "note": "already running"}

        # 소비자 시작 → 루프 스레드 시작
        try:
            self.consumer.start()
        except Exception:
            # consumer 내부에서 예외가 나더라도 루프는 뜨게 함
            pass

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"ok": True, "running": True}


    def stop(self) -> dict:
        if not self._running:
            # consumer.stop() 은 idempotent 하다고 가정
            try:
                self.consumer.stop()
            except Exception:
                pass
            return {"ok": True, "running": False, "note": "already stopped"}

        self._running = False
        # consumer 에게도 종료 신호
        try:
            self.consumer.stop()
        except Exception:
            pass

        # 최대 2초 대기
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        return {"ok": True, "running": False}

    def yolo_select_px(self, x: int, y: int) -> dict:
        self.yolo.select(int(x), int(y))
        return {"ok": True, "mode": "px", "x": int(x), "y": int(y)}

    def yolo_select_norm(self, x: float, y: float) -> dict:
        self.yolo.select_norm(float(x), float(y))
        return {"ok": True, "mode": "norm", "x": float(x), "y": float(y)}

    def yolo_clear(self) -> dict:
        self.yolo.clear_selection()
        return {"ok": True, "cleared": True}

    def status(self) -> dict:
        return {
            "running": self._running,
            "rtsp": RTSP_URL,
            "consumer": self.consumer.status() if hasattr(self.consumer, "status") else {},
            "pipelines": {
                "yolo": self.yolo.status(),
                "stt": self.stt.status(),
                "eye": self.eye.status(),
            },
        }


PIPELINE_MANAGER = PipelineManager()