class STTPipeline:
    def __init__(self):
        self._loaded = False
        # TODO: 예: faster-whisper / vosk 등 초기화


    def process_audio(self, pcm16_bytes, sample_rate):
        if not self._loaded:
            self._load()
            # TODO: 스트리밍 STT 버퍼링/인식
        return None


    def _load(self):
        # TODO: STT 엔진 로드
        self._loaded = True


    def status(self):
        return {"loaded": self._loaded}