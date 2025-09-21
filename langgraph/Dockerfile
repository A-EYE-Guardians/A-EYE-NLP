FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ★ 반드시 포함: thirdparty 전체를 이미지에 복사
COPY thirdparty ./thirdparty
COPY langgraph_app.py ./

EXPOSE 9000
CMD ["uvicorn","langgraph_app:app","--host","0.0.0.0","--port","9000"]
