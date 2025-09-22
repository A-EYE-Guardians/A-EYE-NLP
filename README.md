# Main (Windows, local FastAPI)

이 폴더는 **메인 로컬 API**입니다. 윈도우 PC에서 실행하세요.

## 1) 가상환경 생성 & 패키지 설치
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 환경변수(.env) 설정
- `LANGGRAPH_URL` 에 **맥북 IP**를 넣으세요. 예: `http://192.168.0.23:9100`
- `MINIO_ENDPOINT` 도 맥북 IP: `http://192.168.0.23:9000`
- 토큰은 `MAIN_API_TOKEN`, `LG_API_TOKEN` 양쪽에서 동일하게 맞추세요.

## 3) 서버 실행
```powershell
uvicorn main_app:app --host 0.0.0.0 --port 8000 --reload
```

## 4) 동작 확인
```powershell
python client.py
```
콘솔에 `uploaded source url: ...` 와 `detect ack: ...` 가 출력되면 랭그래프(맥)와 콜백 통신이 됩니다.