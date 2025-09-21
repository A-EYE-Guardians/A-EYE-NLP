# A-EYE API Architecture (Main local + LangGraph in Docker + MinIO)

## 구성
- **Main (local FastAPI)**: 콜백/업로드/트리거 엔드포인트
- **LangGraph (Docker FastAPI)**: 탐지 파이프라인 시뮬레이터
- **MinIO (Docker)**: 중앙 스토리지(S3 호환) – presigned URL 생성/공유

## 실행 순서
1) Docker 서비스 시작 (langgraph, minio)
```bash
cd compose
docker compose up -d --build
```

2) Main 로컬 서버 실행
```bash
cd ../main-local
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main_app:app --reload --port 8000
```

3) 데모 실행
```bash
python client.py
```
- 업로드 → presigned URL 생성 → LangGraph /detect 호출 → LangGraph가 Main 콜백으로 ROI 요청/컨펌 요청 → 최종 결과 콜백

## 네트워킹
- LangGraph→Main 콜백 주소는 요청 본문 `callback_base`에서 받습니다.
  - 로컬/맥/윈도우: `http://host.docker.internal:8000/callbacks/{cid}` 권장
  - 본 템플릿은 `.env`의 `CALLBACK_BASE`로 `http://localhost:8000/callbacks`를 사용하고 있으니,
    리눅스 Docker 환경에서는 host IP로 바꿔주세요.

## 토큰
- Main이 노출하는 콜백을 호출할 때는 `MAIN_API_TOKEN`(기본: MAIN_SECRET)을 Bearer로 사용
- Main→LangGraph 호출 시 `LG_API_TOKEN`(기본: LG_SECRET) 사용

## ROI 전달
- 기본: MinIO URL로 전달(presigned)
- 소용량·실험: base64 인라인 응답으로 대체 가능(코드 변경 필요)

## 커스터마이즈 포인트
- 실제 ROI 생성/캡처 로직: `need_roi` 콜백 핸들러 안에서 구현
- 실제 탐지 파이프라인: LangGraph `run_pipeline` 내부 구현
- 스키마: `cid/req_id/in_reply_to` 확장, 에러/타임아웃 처리 추가 권장
