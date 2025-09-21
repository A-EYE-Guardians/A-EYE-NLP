import base64
import numpy as np
from PIL import Image
import io
from openai import OpenAI

client = OpenAI()

import base64
from mimetypes import guess_type

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Example usage
image_path = '<path_to_image>'
data_url = local_image_to_data_url(image_path)
print("Data URL:", data_url)

def ndarray_to_base64(ndarr: np.ndarray, format: str = "JPEG") -> str:
    """
    NumPy ndarray 이미지를 base64 문자열로 변환
    GPT API에서 사용할 수 있는 data:image/...;base64, 형식으로 반환
    """
    # ndarray → PIL Image
    image = Image.fromarray(ndarr.astype("uint8"))

    # 메모리 버퍼에 저장
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    # base64 인코딩
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def analyze_image_with_gpt(ndarr: np.ndarray, prompt: str):
    """
    GPT-4o mini 멀티모달 API로 ndarray 이미지를 분석
    """
    # ndarray → base64
    img_base64 = ndarray_to_base64(ndarr)

    # API 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base64
                        }
                    }
                ],
            }
        ],
    )

    '''
    # MS 최신 예제(https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision?utm_source=chatgpt.com&tabs=rest)
    {
    "messages": [ 
        {
            "role": "system", 
            "content": "You are a helpful assistant." 
        },
        {
            "role": "user", 
            "content": [
	            {
	                "type": "text",
	                "text": "Describe this picture:"
	            },
	            {
	                "type": "image_url",
	                "image_url": {
                        "url": "<image URL>",
                        # 그냥 base64 기반으로 통일
                        #"url": "data:image/jpeg;base64,<your_image_data>",
                        # 저화질로 통일
                        "detail": "low" # auto(입력 크기 따라 지정), high가 있음
                    }
                } 
           ] 
        }
    ],
    "max_tokens": 100, 
    "stream": false 
}
    '''

    return response.choices[0].message["content"]


# 🔹 사용 예시
if __name__ == "__main__":
    # 가짜 ndarray 생성 (예: 100x100 RGB)
    test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    result = analyze_image_with_gpt(test_array, "이 이미지에 있는 물체를 설명해줘")
    print(result)
