# agent/llm_agent.py
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def llm_chat(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=512):
    if client is None:
        return "[ERROR] OPENAI_API_KEY not set"
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# Actor: 초기 답변 생성 (반환: structured dict-like text)
def actor_answer(question: str) -> str:
    prompt = [
        {"role":"system", "content": "당신은 전문 문서 분석가이자 요약가입니다."},
        {"role":"user", "content":
            f"질문: {question}\n\n"
            "1) 이 질문에 대해 알고 있으면 간결히 10문장 이내로 답변하라.\n"
            "2) 답변 아래에 반드시 'SEARCH_QUERIES:' 라벨을 붙이고, 해당 답변을 검증하거나 확장하는데 유용한 웹검색 쿼리 1~3개를 한 줄씩 적어라.\n"
            "출력 포맷 예시:\n"
            "답변 텍스트...\n\nSEARCH_QUERIES:\n- query 1\n- query 2\n"
        }
    ]
    return llm_chat(prompt, model="gpt-4o-mini", max_tokens=600)


# Evaluator: actor 결과 평가 (간단한 포맷)
def evaluator_evaluate(actor_output: str, env_feedback: str = None) -> str:
    prompt = [
        {"role":"system","content":"당신은 답변의 정확성, 불일치, 누락을 평가하는 평가자입니다."},
        {"role":"user","content": f"아래 답변을 평가하세요:\n\n{actor_output}\n\n외부피드백:\n{env_feedback or '없음'}\n\n결과를 간단히 기술하라."}
    ]
    return llm_chat(prompt, model="gpt-4o-mini", max_tokens=300)

# Reflector: 평가를 바탕으로 반성문(구조화) 생성
def reflector_make_reflection(actor_output: str, evaluation: str) -> dict:
    prompt = [
        {"role":"system","content":"당신은 자각적 자기반성가입니다. 평가를 보고 '누락', '불필요', '개선제안'을 1-2문장씩 작성하세요."},
        {"role":"user","content": f"답변:\n{actor_output}\n\n평가:\n{evaluation}\n\n출력 포맷: JSON with keys missing, superfluous, suggestion."}
    ]
    raw = llm_chat(prompt, model="gpt-4o-mini", max_tokens=300)
    # best-effort parse JSON
    import json
    try:
        return json.loads(raw)
    except Exception:
        # fallback: wrap as suggestion only
        return {"missing":"", "superfluous":"", "suggestion": raw.strip()}
