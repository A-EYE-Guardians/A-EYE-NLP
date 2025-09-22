# document_action.py
"""
document action handlers using OpenAI GPT for:
- document_summary
- answer_by_document
- thesis_search
- generate_exam_questions (기출문제 생성)
+로 추가
highlight: 중요한 부분만 뽑아 강조 (하이라이트 본 생성)
compare_documents: 두 문서 비교 (유사도, 차이점 요약)
qa_generation: 문서를 기반으로 예상 문제(기출문제 스타일) 생성
translation: 문서 번역 (예: 영어 논문 → 한국어 요약)
timeline_generation: 역사/연구/사건 관련 문서에서 연대표 자동 생성
code_extraction: 기술문서/논문에서 코드 블록만 뽑아서 정리
Relies on processing_document.process_document_for_llm and retrieval utilities.
"""
import os
import sys
import re
from typing import Dict, List
import textwrap
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import docx
from bs4 import BeautifulSoup
import mammoth   # .docx → .html 변환 보조
import mimetypes
from typing import Dict, List
from urllib.parse import urlparse
from util.search_utils import google_search  # 사용 가능한 검색 함수
import logging
DOCUMENT_CACHE = {}
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from loader.processing_document import process_document_for_llm, retrieve_top_k, process_multiple_documents, cosine_sim

DOCUMENT_ACTION_HANDLERS = {}

# ----------------------------
# 기본 설정
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if client is None:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")

def register_document_action(name):
    def decorator(func):
        DOCUMENT_ACTION_HANDLERS[name] = func
        return func
    return decorator

def gpt_query(prompt: str, temperature=0.0, max_tokens=1024):
    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "너는 전문 문서 분석용 AI야."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[GPT ERROR] {e}"

# utils
def safe_truncate(s: str, max_chars=3000):
    return s if len(s) <= max_chars else s[:max_chars-100] + "\n\n[... truncated ...]"

# -------------------------
# GPT helpers
# -------------------------
def gpt_chat(messages: List[Dict], model="gpt-4o-mini", temperature=0.0, max_tokens=512):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[GPT ERROR] {e}"

def summarize_chunks(processed_doc: Dict, max_chars_per_chunk=1500):
    # summarize each chunk, then combine
    chunks = processed_doc.get("chunks", [])
    summaries = []
    for c in chunks:
        text = safe_truncate(c["text"], max_chars=max_chars_per_chunk)
        prompt = [
            {"role":"system","content":"너는 전문 문서 요약가야. 핵심만 간결히 요약해줘."},
            {"role":"user","content": f"다음 텍스트를 2-3문장으로 요약해줘:\n\n{text}"}
        ]
        s = gpt_chat(prompt, max_tokens=200)
        summaries.append(s)
    # combine summaries
    combined = "\n".join(summaries)
    # final synthesis
    prompt2 = [
        {"role":"system","content":"너는 문서 종합 요약가야. 아래 요약들을 읽고 최종 요약을 5문장 내로 작성해줘."},
        {"role":"user","content": combined}
    ]
    final = gpt_chat(prompt2, max_tokens=400)
    return final

# ------------------------
# 문서 로더들
# 2. 확장 가능한 문서 처리 파이프라인 설계
#
# 지원 포맷: .pdf, .docx, .hwp/.hwpx, .txt, .html, .md, .js/.ts (코드도 텍스트화)
#
# 처리 단계:
#
# 포맷 감지 (mimetypes or 확장자)
#
# 전처리 (HTML → BeautifulSoup, DOCX → python-docx, HWP/HWPX → pyhwp/olefile 기반, PDF → pypdf/OCR)
#
# 텍스트 클린업 (줄바꿈, 공백, 특수문자 정리)
#
# chunking (길면 분할) + GPT action 전달
# ------------------------
def load_pdf_text(file_path: str) -> str:
    """텍스트 PDF 우선 → OCR 백업"""
    try:
        import PyPDF2
        text = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
        joined = "\n".join(text).strip()
        if joined:
            return joined
    except Exception:
        pass
    # OCR fallback
    return pdf_ocr_to_text(file_path)

def pdf_ocr_to_text(pdf_path: str) -> str:
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        text = []
        for page in pages:
            text.append(pytesseract.image_to_string(page, lang="kor+eng"))
        return "\n".join(text)
    except Exception as e:
        return f"[ERROR] OCR 실패: {e}"

def load_docx_text(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"[ERROR] DOCX 로드 실패: {e}"

def load_txt_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] TXT 로드 실패: {e}"

def load_html_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style"]): s.decompose()
        text = soup.get_text(separator="\n")
        return re.sub(r"\n+", "\n", text).strip()
    except Exception as e:
        return f"[ERROR] HTML 로드 실패: {e}"

def load_document(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf_text(file_path)
    elif ext in [".docx"]:
        return load_docx_text(file_path)
    elif ext in [".txt", ".md", ".js", ".ts", ".py"]:
        return load_txt_text(file_path)
    elif ext in [".html", ".htm"]:
        return load_html_text(file_path)
    else:
        return f"[WARN] 지원되지 않는 포맷: {ext}"

# -------------------------
# Action handlers
# -------------------------
@register_document_action("document_summary")
def _document_summary_action(data: dict):
    urls = data.get("urls", [])
    if not urls:
        return "[ERROR] 요약할 문서 URL 없음"
    docs = process_multiple_documents(urls, from_url=True, do_embed=True)
    text = docs[0].get("raw_text", "")
    if not text:
        return "[ERROR] 문서 내용 없음"
    prompt = [
        {"role": "system", "content": "너는 전문 문서 요약가야."},
        {"role": "user", "content": f"다음 문서를 5문장 이내로 요약해줘:\n\n{safe_truncate(text)}"}
    ]
    return gpt_chat(prompt, max_tokens=500)

@register_document_action("answer_by_document")
def _answer_by_document_action(data: dict):
    urls = data.get("urls", [])
    question = data.get("question") or data.get("command")
    if not urls or not question:
        return "[ERROR] URL 또는 질문 없음"
    docs = process_multiple_documents(urls, from_url=True, do_embed=True)
    hits_texts = [d.get("raw_text", "")[:1500] for d in docs if d.get("raw_text")]
    context = "\n\n".join(hits_texts)
    prompt = [
        {"role": "system", "content": "너는 문서 기반 질의응답 도우미야."},
        {"role": "user", "content": f"문서 발췌:\n{safe_truncate(context)}\n\n질문: {question}"}
    ]
    return gpt_chat(prompt, max_tokens=600)

def _search_papers_action(data: dict):
    # 이미 main.py에서 검색됨
    return {"urls": data.get("urls", [])}

'''
@register_document_action("thesis_search")
def _thesis_search_action(query: str, num_results=5, do_embed=True):
    """
    간단한 논문 검색: 구글 크롤러나 serapi에서 얻은 URL 리스트를 넘겨주면
    각 URL의 텍스트를 가져와 요약 반환.
    For production, integrate with arXiv API / Semantic Scholar API.
    """
    # Prefer arxiv API if query looks academic
    # Here we'll do a simple google search util (user already has google_search)
    from search_utils import google_search
    results = google_search(query, num_results, sleep_sec=1)
    summaries = []
    for r in results:
        url = r.get("link")
        if not url:
            continue
        processed = process_document_for_llm(url, from_url=True, do_embed=False)
        text = processed.get("raw_text","")
        if not text:
            continue
        short = safe_truncate(text, max_chars=1500)
        prompt = [
            {"role":"system","content":"너는 논문 요약 전문가야."},
            {"role":"user","content": f"다음 기사/논문(일부)을 읽고 2-3문장으로 핵심을 요약해라:\n\n{short}"}
        ]
        s = gpt_chat(prompt, max_tokens=250)
        summaries.append({"title": r.get("title"), "url": url, "summary": s})
    return summaries
'''

@register_document_action("qa_generation")
def _qa_generation_action(data: dict):
    urls = data.get("urls", [])
    n = data.get("n", 5)
    docs = process_multiple_documents(urls, from_url=True, do_embed=False)
    text = docs[0].get("raw_text", "") if docs else ""
    if not text:
        return "[ERROR] 문서 없음"
    prompt = [
        {"role": "system", "content": "너는 교육용 문제를 만드는 교사야."},
        {"role": "user", "content": f"다음 텍스트를 바탕으로 예상 문제 {n}개와 정답을 만들어줘:\n\n{text[:3000]}"}
    ]
    return gpt_chat(prompt, max_tokens=800)

@register_document_action("highlight")
def _highlight_action(data: dict):
    urls = data.get("urls", [])
    top_n = data.get("n", 10)
    docs = process_multiple_documents(urls, from_url=True, do_embed=False)
    text = docs[0].get("raw_text", "") if docs else ""
    if not text:
        return "[ERROR] 문서 없음"
    prompt = [
        {"role": "system", "content": "너는 중요한 문장을 뽑아 강조하는 편집자야."},
        {"role": "user", "content": f"다음 텍스트에서 핵심 문장 {top_n}개를 골라줘:\n\n{text[:3000]}"}
    ]
    return gpt_chat(prompt, max_tokens=500)


# -------------------------
# compare_documents (multi-doc) - 수정
# -------------------------
@register_document_action("compare_documents")
def _compare_documents_action(data: dict):
    """
    두 개 이상의 문서를 비교.
    data 구조 통일:
    {
        "documents": [
            {"source": "url_or_path1", "from_url": True},
            {"source": "url_or_path2", "from_url": True}
        ],
        "command": "비교 명령문/설명 (옵션)"
    }
    """
    urls = data.get("documents", [])
    if len(urls) < 2:
        return "[ERROR] 최소 2개 이상의 문서 필요"
    docs = process_multiple_documents(urls, from_url=True, do_embed=True)
    if len(docs) < 2:
        return "[ERROR] 문서 로드 실패"
    results = []
    for i in range(len(docs) - 1):
        for j in range(i + 1, len(docs)):
            t1 = docs[i].get("raw_text", "")[:3000]
            t2 = docs[j].get("raw_text", "")[:3000]
            emb1 = np.mean([c["embedding"] for c in docs[i].get("chunks", []) if c.get("embedding")], axis=0)
            emb2 = np.mean([c["embedding"] for c in docs[j].get("chunks", []) if c.get("embedding")], axis=0)
            sim = float(cosine_sim(emb1, emb2)) * 100 if emb1 is not None and emb2 is not None else 0
            prompt = [
                {"role": "system", "content": "너는 문서 비교 전문가야."},
                {"role": "user", "content": f"문서1:\n{t1}\n\n문서2:\n{t2}\n\n공통점과 차이점을 정리해줘."}
            ]
            summary = gpt_chat(prompt, max_tokens=600)
            results.append({
                "doc1": urls[i],
                "doc2": urls[j],
                "similarity": sim,
                "summary": summary
            })
    return results


@register_document_action("translation")
def _translation_action(data: dict):
    urls = data.get("urls", [])
    lang = data.get("language", "한국어")
    docs = process_multiple_documents(urls, from_url=True, do_embed=False)
    text = docs[0].get("raw_text", "") if docs else ""
    if not text:
        return "[ERROR] 문서 없음"
    prompt = f"다음 문서를 {lang}로 번역해줘:\n\n{text[:3000]}"
    return gpt_chat([{"role": "user", "content": prompt}], max_tokens=800)

@register_document_action("timeline_generation")
def _timeline_generation(data: dict):
    urls = data.get("urls", [])
    docs = process_multiple_documents(urls, from_url=True, do_embed=False)
    text = docs[0].get("raw_text", "") if docs else ""
    if not text:
        return "[ERROR] 문서 없음"
    prompt = [
        {"role": "system", "content": "너는 역사 및 사건을 시간순으로 정리하는 전문가야."},
        {"role": "user", "content": f"다음 문서를 기반으로 연대표를 작성해줘:\n\n{text[:3000]}"}
    ]
    return gpt_chat(prompt, max_tokens=600)

@register_document_action("code_extraction")
def _code_extraction_action(data: dict):
    urls = data.get("urls", [])
    docs = process_multiple_documents(urls, from_url=True, do_embed=False)
    text = docs[0].get("raw_text", "") if docs else ""
    if not text:
        return "[ERROR] 문서 없음"
    prompt = [
        {"role": "system", "content": "너는 기술문서에서 코드 블록을 추출하는 도우미야."},
        {"role": "user", "content": f"다음 문서에서 코드 블록을 추출해줘:\n\n{text[:3000]}"}
    ]
    return gpt_chat(prompt, max_tokens=800)

# -------------------------
# Dispatcher for integration with main.py: + command 기반 파라미터 추출
# -------------------------
DOCUMENT_ACTION_MAP = {
    "document_summary": _document_summary_action,
    "answer_by_document": _answer_by_document_action,
    "thesis_search": _search_papers_action,
    "qa_generation": _qa_generation_action,
    "highlight": _highlight_action,
    "compare_documents": _compare_documents_action,
    "translation": _translation_action,
    "timeline_generation": _timeline_generation,
    "code_extraction": _code_extraction_action
}

def handle_document_action(command_text: str, sources: list = None, top_k: int = 3):
    """
    command_text: 사용자 질의
    sources: URL 리스트. 없으면 웹 검색 후 수집
    top_k: 반환할 청크 개수
    """
    final_texts = []

    # 1) sources가 없으면 웹 검색 수행
    if not sources:
        search_results = google_search(command_text, num_results=5)
        sources = [r["link"] for r in search_results]

    # 2) 문서 처리 + embedding
    new_sources = [s for s in sources if s not in DOCUMENT_CACHE]
    if new_sources:
        processed_list = process_multiple_documents(new_sources, do_embed=True)
        for src, proc in zip(new_sources, processed_list):
            DOCUMENT_CACHE[src] = proc

    # 3) query 기준 top-k 청크 추출
    for src in sources:
        proc_doc = DOCUMENT_CACHE.get(src)
        if proc_doc:
            chunks = retrieve_top_k(command_text, proc_doc, top_k=top_k)
            final_texts.extend([c["text"] for c in chunks])

    return "\n\n".join(final_texts) if final_texts else "[SYSTEM] 관련 문서 내용 없음"

'''
def handle_document_action(processed: Dict):
    
    # processed: process_document_for_llm 결과 + 'action', 'params', 'command' 포함
    
    action = processed.get("action")
    params = processed.get("params", {}) or {}
    command = processed.get("command", "")

    if action not in DOCUMENT_ACTION_MAP:
        return f"[ERROR] unknown document action: {action}"
    fn = DOCUMENT_ACTION_MAP[action]

    # 개수/출력 관련 추론
    def parse_int_from_command(cmd: str, default: int):
        m = re.search(r"(\d+)", cmd or "")
        return int(m.group(1)) if m else default

    if action == "document_summary":
        src = params.get("url") or processed.get("document_path") or processed.get("source") or command
        return fn(src, from_url=True, do_embed=True)
    if action == "answer_by_document":
        src = params.get("url") or processed.get("document_path") or processed.get("source")
        question = params.get("question") or command
        return fn(src, question, from_url=True, do_embed=True)
    if action == "thesis_search":
        query = params.get("query") or command
        n = parse_int_from_command(command, params.get("num", 5))
        return fn(query, num_results=n)
    if action == "qa_generation":
        src = params.get("url") or processed.get("document_path") or processed.get("source")
        n = parse_int_from_command(command, params.get("n", 5))
        return fn(src, from_url=True, n_questions=n)
    if action == "highlight":
        src = params.get("url") or processed.get("document_path") or processed.get("source")
        top_n = parse_int_from_command(command, params.get("top_n", 10))
        return fn(src, from_url=True, top_n=top_n)
    if action == "compare_documents":
        # 두 문서 소스 확인
        src1 = params.get("document_text1_source") or processed.get("source1")
        src2 = params.get("document_text2_source") or processed.get("source2")

        if not src1 or not src2:
            return "[ERROR] 비교할 두 문서 소스를 제공해야 합니다."

        # 두 문서 처리 (process_multiple_documents)
        processed_docs = process_multiple_documents([src1, src2], from_url=True, do_embed=True)

        # 상태 체크
        for idx, doc in enumerate(processed_docs):
            if doc["status"] != "ok":
                return f"[ERROR] 문서 {idx + 1} 처리 실패: {doc['error_msg']}"

        # compare_documents 실행
        return _compare_documents_action({
            "document_text1": processed_docs[0]["raw_text"],
            "document_text2": processed_docs[1]["raw_text"]
        }, command)

    if action == "translation":
        src = params.get("url") or processed.get("document_path") or processed.get("source")
        return fn(src, from_url=True, do_embed=True)
    if action == "timeline_generation":
        src = params.get("url") or processed.get("document_path") or processed.get("source")
        return fn(src, from_url=True, do_embed=True)
    if action == "code_extraction":
        src = params.get("url") or processed.get("document_path") or processed.get("source")
        return fn(src, from_url=True, do_embed=True)
'''