import av
import threading
import queue
import numpy as np


class RTSPConsumer:
    def __init__(self, url: str):
        self.url = url
        self._thread = None
        self._running = False
        self._q = queue.Queue(maxsize=100)
        self._status = {'video_frames': 0, 'audio_frames': 0}


    def start(self):   
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()


    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


    def status(self):
        return dict(self._status)


    def iter_frames(self):
        while self._running:
            try:
                item = self._q.get(timeout=1)
                yield item
            except queue.Empty:
                continue


    def _run(self):
        # PyAV를 이용한 RTSP Demux (비디오+오디오)
        # 옵션: TCP 선호, 타임아웃 등
        container = av.open(self.url, options={'rtsp_flags': 'prefer_tcp', 'stimeout': '5000000'})
        streams = [s for s in container.streams if s.type in ('video', 'audio')]
        for packet in container.demux(streams):
            if not self._running:
                break
            for frame in packet.decode():
                if not self._running:
                    break
                if isinstance(frame, av.VideoFrame):
                    img = frame.to_ndarray(format='bgr24')
                    self._status['video_frames'] += 1
                    self._put(('video', img))
                elif isinstance(frame, av.AudioFrame):
                # 오디오를 PCM16으로 변환
                    pcm = frame.to_ndarray().astype(np.int16).tobytes()
                    sr = frame.sample_rate
                    self._status['audio_frames'] += 1
                    self._put(('audio', (pcm, sr)))


    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
        # 가장 오래된 프레임 드랍(실시간성 유지)
            try:
                self._q.get_nowait()
                self._q.put_nowait(item)
            except queue.Empty:
                pass