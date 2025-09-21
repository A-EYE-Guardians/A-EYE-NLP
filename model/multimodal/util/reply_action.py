import json
import time
from openai import OpenAI
from .search_utils import google_search

client = OpenAI()
conversation_history = []


def reply_with_memory(user_text: str) -> str:
    """단순 대화/질문에 답변. 필요 시 검색 + 요약."""
    global conversation_history

    conversation_history.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 AR 글래스 비서 챗봇이야. "
                                          "필요하다면 웹 검색을 요청할 수 있어."}
        ] + conversation_history,
        temperature=0.7,
        max_tokens=1024,
        functions=[
            {
                "name": "search_web",
                "description": "사용자 질문에 답하기 위해 인터넷 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "검색 키워드"}
                    },
                    "required": ["query"]
                }
            }
        ],
        function_call="auto"
    )

    msg = resp.choices[0].message
    if msg.function_call:
        # GPT가 검색을 요구하는 경우
        args = json.loads(msg.function_call.arguments)
        query = args.get("query")
        search_results = google_search(query, num_results=5)
        snippets = [f"{r['title']}: {r['snippet']}" for r in search_results]
        search_result_text = "\n".join(snippets)

        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 AR 글래스 비서 챗봇이야."},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": None, "function_call": msg.function_call},
                {"role": "function", "name": "search_web", "content": search_result_text},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        reply = resp2.choices[0].message.content.strip()
    else:
        reply = msg.content.strip()

    conversation_history.append({"role": "assistant", "content": reply})

    # 300자 초과 시 요약
    if len(reply) > 300:
        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "긴 답변을 300자 내외로 요약해 음성 출력용으로 간결히 만들어라."},
                {"role": "user", "content": reply}
            ],
            temperature=0
        )
        reply = summary.choices[0].message.content.strip()

    return reply
