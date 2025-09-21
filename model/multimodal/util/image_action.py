# image_action.py
import torch
from pathlib import Path
from PIL import Image
import webbrowser
import numpy as np
import pytesseract
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import cv2

# CPU 설정
device = torch.device("cpu")

# handlers.py
ACTION_REGISTRY = {}

def register_action(name):
    def decorator(func):
        ACTION_REGISTRY[name] = func
        return func
    return decorator

# BLIP-2 모델 로드 (작은 사이즈, CPU용)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
model.eval()  # inference only

def tensor_to_pil(image_tensor):
    img = (image_tensor.permute(1,2,0).numpy() * 255).astype("uint8")
    return Image.fromarray(img)

# YOLO World로 사전학습된 객체인식 모듈로 이미지 내 객체 인식 및 라벨링 후 llm로 해당 객체에 대한 설명이 필요할 때 사용되는 모듈
def _object_info(image_tensor, command: str):
    img = tensor_to_pil(image_tensor)
    inputs = processor(images=img, text=command, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    result = processor.decode(output_ids[0], skip_special_tokens=True)
    return result

# pytesseract pakage로 이미지 내 문자 ocr이 필요할 때 사용하는 모듈
def _text_recognition(image_tensor, command: str):
    img = tensor_to_pil(image_tensor)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    return f"OCR 결과: {text.strip()}"

# 제공된 이미지를 바탕으로 이용자가 목적지까지 가는 길에 향해야 할 방향을 후처리해 알려 주는 모듈로 가정
def _navigate_image(image_tensor, command: str):
    img = tensor_to_pil(image_tensor)
    inputs = processor(images=img, text=command, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    result = processor.decode(output_ids[0], skip_special_tokens=True)
    return result

# pyzbar pakage로 QR 코드 스캔이 필요할 때 사용하는 모듈
def _scan_code(image_tensor, command: str):
    # 1. image_tensor를 OpenCV가 사용하는 numpy 배열로 변환
    pil_image = tensor_to_pil(image_tensor)
    cv_image = np.array(pil_image)

    # 2. QR 코드 탐지기 객체 생성
    detector = cv2.QRCodeDetector()

    # 3. QR 코드 탐지 및 디코딩
    data, _, _ = detector.detectAndDecode(cv_image)

    if data:
        # data는 탐지된 QR 코드의 문자열
        webbrowser.open(data)
        return f"QR 코드 인식 성공: {data}"
    else:
        return "QR 코드 인식 실패"

# BLIP-2로 이미지 설명이 필요할 때 사용하는 모듈
def _control_hw(image_tensor, command: str):
    img = tensor_to_pil(image_tensor)
    # 실제 하드웨어 없으므로 BLIP-2로 이미지 설명
    inputs = processor(images=img, text=command, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    result = processor.decode(output_ids[0], skip_special_tokens=True)
    return result

# 핸들러
IMAGE_ACTION_HANDLERS = {
    "object_info": _object_info,
    "text_recognition": _text_recognition,
    "navigate_image": _navigate_image,
    "scan_code": _scan_code,
    "control_hw": _control_hw
}

def handle_image_action(processed: dict, save_path: Path):
    act = processed.get("action")
    image_tensor = processed.get("image_tensor")
    command = processed.get("command")
    handler = IMAGE_ACTION_HANDLERS.get(act)
    if not handler:
        return f"[ERROR] 미정의 이미지 action: {act}"
    return handler(image_tensor, command)
