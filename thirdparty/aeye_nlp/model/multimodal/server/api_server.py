# api_server.py (핵심 엔드포인트)
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uuid, json, os
from pathlib import Path
import boto3  # optional MinIO/S3

app = FastAPI()
STORAGE_DIR = Path("./uploads"); STORAGE_DIR.mkdir(exist_ok=True)

# fake enqueue function - replace with Redis/Kafka/NATS publish
def enqueue_job(job_payload: dict):
    # push to Redis stream / RabbitMQ etc.
    print("ENQUEUE", job_payload)

@app.post("/ingest/image")
async def ingest_image(device_id: str = Form(...), action_suggest: str = Form(...),
                       metadata: str = Form(None), file: UploadFile = File(...)):
    # save file
    job_id = str(uuid.uuid4())
    save_path = STORAGE_DIR / f"{job_id}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    meta = json.loads(metadata) if metadata else {}
    job = {
      "job_id": job_id,
      "device_id": device_id,
      "action": action_suggest,
      "file_path": str(save_path),
      "metadata": meta
    }
    enqueue_job(job)
    return JSONResponse({"job_id": job_id, "status": "queued"})

@app.post("/ingest/audio")
async def ingest_audio(device_id: str = Form(...), action_suggest: str = Form(...),
                       file: UploadFile = File(...)):
    # similar pattern
    job_id = str(uuid.uuid4())
    save_path = STORAGE_DIR / f"{job_id}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    job = {"job_id": job_id, "device_id": device_id, "action": action_suggest, "file_path": str(save_path)}
    enqueue_job(job)
    return JSONResponse({"job_id": job_id, "status": "queued"})

@app.get("/result/{job_id}")
def get_result(job_id: str):
    # read DB for result row (pseudo)
    # return JSON with status & outputs (text, s3 links, etc.)
    return JSONResponse({"job_id": job_id, "status": "done", "text": "예시 결과"})
