class EyePipeline:
    def __init__(self):
        self._loaded = False
        # TODO: MediaPipe/커스텀 모델 등 초기화


    def process_frame(self, bgr):
        if not self._loaded:
            self._load()
        #   TODO: 시선 추정/시각화 등
        return None


    def _load(self):
        self._loaded = True


    def status(self):
        return {"loaded": self._loaded}