# worker.py (sketch)
import time, json
from pathlib import Path
import torch
from PIL import Image
# import model libs (BLIP/CLIP/your LLM wrapper)

def process_job(job):
    action = job["action"]
    file_path = job["file_path"]
    # 1) load file
    img = Image.open(file_path).convert("RGB")
    # 2) preproc -> tensor
    # 3) handle actions
    if action == "object_info":
        # run blip/clip model -> description
        text = run_blip_object_info(img, job.get("metadata", {}))
    elif action == "text_recognition":
        text = run_ocr(img)
    # 4) save result to DB or S3
    save_result(job["job_id"], {"text": text})
    # 5) optionally callback edge URL
    if job["metadata"].get("callback_url"):
        requests.post(job["metadata"]["callback_url"], json={"job_id": job["job_id"], "result": text})

def worker_loop():
    while True:
        job = poll_queue()
        if job:
            try:
                process_job(job)
            except Exception as e:
                # log, retry logic
                print("job failed", e)
        else:
            time.sleep(0.2)
