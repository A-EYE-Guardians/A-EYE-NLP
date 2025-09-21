# processing_image.py
import json
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import requests
import os
import base64
from PIL import Image
import io

def ndarray_to_base64(ndarr: np.ndarray, format: str = "JPEG") -> str:
    """
    NumPy ndarray 이미지를 base64 문자열로 변환
    GPT API에서 사용할 수 있는 data:image/...;base64, 형식으로 반환
    """
    image = Image.fromarray(ndarr.astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

SAMPLE_IMAGE_PATH = "./loader/sample.jpg"  # 예외 처리용 샘플 이미지
SERVER_URL = "http://localhost:8000/get_image"  # 이미지 요청 서버 예시

def fetch_image_from_server(command: str, action: dict) -> Image.Image:
    """
    command + action을 서버에 보내서 이미지 받아오기
    """
    try:
        payload = {"command": command, "action": action}
        response = requests.post(SERVER_URL, json=payload, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[경고] 서버 이미지 요청 실패({e}), 샘플 이미지 사용")
        return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")

'''
def load_image_from_url(url: str) -> Image.Image:
    """URL에서 이미지를 다운로드하여 PIL.Image로 반환"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[경고] 이미지 로드 실패({e}), 샘플 이미지 사용")
        img = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    return img
'''

def crop_roi(img: Image.Image, bbox: dict) -> Image.Image:
    """bbox 기반 ROI crop 수행"""
    try:
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        return img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"[경고] ROI crop 실패({e}), 전체 이미지 사용")
        return img

def image_to_tensor(img: Image.Image, size=(224, 224)) -> torch.Tensor:
    """PIL 이미지 → torch.Tensor (C,H,W), 0~1 정규화"""
    img_resized = img.resize(size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0  # HWC
    tensor = torch.from_numpy(img_array).permute(2,0,1)  # CHW
    return tensor

def process_image(action: dict, command: str) -> dict:
    """
    JSON + command → 멀티모달 연산 가능 tensor 반환
    JSON 예시:
    {
        "caption": "빨간 운동화",
        "url": "http://example.com/sample.jpg",
        "bbox": {"x1":50,"y1":40,"x2":200,"y2":180}  # optional
    }
    """
    caption = action.get("caption", "No caption")
    #url = action.get("url", None)
    bbox = action.get("bbox", None)

    # URL 없으면 샘플 이미지 사용
    #if not url:
     #   print("[경고] URL 없음, 샘플 이미지 사용")
      #  img = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    #else:
    img = fetch_image_from_server(command, action)

    # ROI crop: bbox 없으면 전체 이미지 사용
    if bbox:
        roi_img = crop_roi(img, bbox)
    else:
        print("[경고] bbox 없음, 전체 이미지 사용")
        roi_img = img

    # tensor 변환
    image_tensor = image_to_tensor(roi_img)

    return {
        "command": command,
        "caption": caption,
        "bbox": bbox,
        "image_tensor": image_tensor,
        "action": action.get("action")  # 여기에 액션 이름 추가
        # 예전: "url": url if url else SAMPLE_IMAGE_PATH
        #"original_image": img  # 필요 시 원본 PIL 이미지도 반환
    }

'''
{
    "command": command,             # STT로 받은 명령어
    "caption": caption,             # JSON에 있던 caption (없으면 "No caption")
    "bbox": bbox,                   # JSON에 있던 bbox (없으면 None)
    "image_tensor": image_tensor,   # ROI crop 후 torch.Tensor (멀티모달 모델 입력용)
    "url": url_or_sample_path        # 원본 URL, 없으면 샘플 이미지 경로
}
앞단(STT + 객체탐지 서버)에서 받은 command + JSON을 → LLM/멀티모달 모델이 이해 가능한 이미지 tensor 형태로 변환해 반환하는 역할만 수행.

- 파일 저장 없음 → 메모리에서 tensor 바로 연산 가능

- ROI가 없거나 URL이 없으면 예외 처리 후 fallback

- image_action.py에서는 이 tensor만 받아서 액션 수행 가능
'''

if __name__ == "__main__":
    import torch
    import numpy as np
    from pathlib import Path

    SAMPLE_IMAGE_PATH = Path("sample.jpg")  # 샘플 이미지 경로
    SERVER_URL = "http://localhost:8000/fetch_image"  # 테스트용 서버 URL

    # 예시 command + action
    command = "테스트 이미지 처리"
    action = {
        "caption": "테스트 이미지",
        "bbox": {"x1": 50, "y1": 40, "x2": 200, "y2": 180},
        "action": "process_image"
    }

    # process_image 실행
    result = process_image(action, command)
    print("Caption:", result["caption"])
    print("BBox:", result["bbox"])
    print("Action:", result["action"])
    print("Tensor shape:", result["image_tensor"].shape)

    # image_file_to_ndarray로 ndarray 변환 테스트
    ndarr = np.array(result["image_tensor"].permute(1, 2, 0))  # CHW → HWC
    print("NDArray shape:", ndarr.shape)
    print("NDArray dtype:", ndarr.dtype)
    print("NDArray sample pixels:", ndarr[0:5, 0:5, :])

