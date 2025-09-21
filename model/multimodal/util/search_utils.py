# search_utils.py
import requests
from bs4 import BeautifulSoup
import time

def google_search(query: str, num_results=5, sleep_sec=2):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    url = f"https://www.google.com/search?q={query}&hl=ko"
    resp = requests.get(url, headers=headers, timeout=5)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for g in soup.select("div.g")[:num_results]:
        title = g.select_one("h3")
        snippet = g.select_one("span.aCOpRe, div.VwiC3b")
        link_tag = g.select_one("a")

        if title and link_tag:
            results.append({
                "title": title.get_text(),
                "snippet": snippet.get_text() if snippet else "",
                "link": link_tag["href"]
            })

    time.sleep(sleep_sec)  # 과도한 요청 방지
    return results
