# LangGraph (Mac, Docker)

이 폴더는 **맥북에서 Docker**로 구동하는 랭그래프 API입니다.
- `agent/` 폴더에 **A‑EYE‑Back의 에이전트 툴스**(누락됐던 부분)를 포함했습니다.
- `/agent/*` 엔드포인트로 바로 호출할 수 있도록 FastAPI 라우터를 추가했습니다.

## 1) 빌드 & 실행 (Mac)
```bash
cd langgraph_docker
docker compose -f compose.yaml up -d --build
# 이후 http://localhost:9100/docs 로 API 확인
```

## 2) 메인(Windows)와 연결
- 윈도우 `.env`에서 `LANGGRAPH_URL` 을 `http://<맥IP>:9100` 으로 변경
- 윈도우 `.env`에서 `MINIO_ENDPOINT` 를 `http://<맥IP>:9000` 으로 변경
- 토큰 `MAIN_API_TOKEN` / `LG_API_TOKEN` 을 각각 `MAIN_SECRET` / `LG_SECRET` 으로 맞추세요.
  - 맥의 `compose.yaml`에 있는 `MAIN_TOKEN`, `LG_TOKEN`과 **일치**해야 합니다.

## 3) Agent 도구 사용 (옵션)
에이전트 툴은 Redis 스트림을 사용합니다. (compose에 redis 포함)
```bash
# 상태 확인
curl http://localhost:9100/agent/status

# 파이프라인 시작/중지
curl -X POST http://localhost:9100/agent/start
curl -X POST http://localhost:9100/agent/stop

# ROI 최신 썸네일 (base64 data URL)
curl http://localhost:9100/agent/roi
```
> 참고: 현재 기본 ROI 경로는 MinIO를 사용합니다만, `agent/tools.py`는 `/shared/uploads`를 참조합니다.
> MinIO 대신 호스트 디렉터리 공유로 전환하려면 `compose.yaml`의 주석 부분을 활성화하고
> `SHARED_UPLOADS=/shared/uploads` 경로로 ROI를 저장하도록 워커 코드를 조정하세요.
