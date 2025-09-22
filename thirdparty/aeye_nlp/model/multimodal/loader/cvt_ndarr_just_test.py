# utils/image_utils.py
import numpy as np
from PIL import Image
import cv2
from fontTools.misc.cython import returns
from pdf2image import convert_from_path
from pathlib import Path
from typing import List, Union

def image_file_to_ndarray(path: Union[str, Path]) -> np.ndarray:
    """
    이미지 파일을 읽어서 ndarray 반환.
    지원 포맷: PNG, JPG, JPEG, BMP, GIF
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)

def resize_ndarray(img_array: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    ndarray 크기 조정 (OpenCV 사용)
    """
    h, w = img_array.shape[:2]
    if width and not height:
        ratio = width / w
        height = int(h * ratio)
    elif height and not width:
        ratio = height / h
        width = int(w * ratio)
    elif not width and not height:
        return img_array
    resized = cv2.resize(img_array, (width, height))
    return resized

# --------------------------
# 테스트용 샘플 코드
# --------------------------
if __name__ == "__main__":
    img_path = "sample.jpg"
    pdf_path = "sample.pdf"

    img_arr = image_file_to_ndarray(img_path)
    print("Image ndarray shape:", img_arr.shape)
