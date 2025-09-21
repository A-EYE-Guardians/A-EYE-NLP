# agent/tools.py
from typing import List, Dict
import os, sys, re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from loader.processing_document import process_document_for_llm, process_multiple_documents
from util.search_utils import google_search

def run_web_search(queries: List[str], num_results=5) -> List[Dict]:
    results = []
    for q in queries:
        rows = google_search(q, num_results)  # 기존 함수 재사용
        results.append({"query": q, "results": rows})
    return results

def fetch_and_process_urls(urls: List[str], do_embed=False):
    processed = []
    for u in urls:
        p = process_document_for_llm(u, from_url=True, do_embed=do_embed)
        processed.append({"url": u, "processed": p})
    return processed
