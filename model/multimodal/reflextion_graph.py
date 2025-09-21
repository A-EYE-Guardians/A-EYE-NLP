# agent/reflexion_graph.py (수정본)
import time
import json
from agent.llm_agent import actor_answer, evaluator_evaluate, reflector_make_reflection, llm_chat
from agent.tools import run_web_search, fetch_and_process_urls
from agent.reflection import Trajectory, append_reflection
MAX_ITER = 3

def parse_search_queries_from_actor(actor_text: str):
    lines = actor_text.splitlines()
    queries = []
    start = False
    for ln in lines:
        if ln.strip().upper().startswith("SEARCH_QUERIES"):
            start = True
            continue
        if start:
            if not ln.strip():
                break
            queries.append(ln.strip().lstrip("-").strip())
    return queries

def generate_search_queries_fallback(command: str, n=3):
    prompt = [
        {"role":"system","content":"당신은 검색 쿼리 생성기입니다. 사용자의 질문을 보고, 정확하고 간결한 웹검색 쿼리 1~3개를 한 줄씩 제안하세요."},
        {"role":"user","content": f"사용자 질문: {command}\n\n제안할 검색어를 한 줄마다 하나씩 출력하세요."}
    ]
    try:
        resp = llm_chat(prompt, model="gpt-4o-mini", max_tokens=150)
        # split lines, filter empties
        qlines = [l.strip("-• \t") for l in resp.splitlines() if l.strip()]
        return qlines[:n]
    except Exception:
        # fallback: simple keyword split
        kws = command.replace(",", " ").split()
        return [" ".join(kws[:6])]

def run_reflexion_pipeline(command: str, env_feedback: str = None, force_query_generation: bool = False):
    """
    Returns:
      final (str), traj (Trajectory), evidence (dict)
    evidence: {
       "queries": [...],
       "search_results": [...],  # raw web-search results
       "processed_docs": [ {"url":..., "raw_text_len": int, "short": "..."} , ... ]
    }
    """
    traj = Trajectory(timestamp=time.time(), command=command, actor_output="", tool_outputs=[], evaluation=None, reflection=None)

    # 1) Actor
    actor_out = actor_answer(command)
    traj.actor_output = actor_out

    # 2) try parse queries from actor output
    queries = parse_search_queries_from_actor(actor_out)

    # 2b) fallback: if none found or forced, generate via LLM
    if not queries or force_query_generation:
        try:
            queries = generate_search_queries_fallback(command, n=3)
        except Exception:
            queries = []

    tool_res = None
    processed = []
    if queries:
        # web search for each query
        try:
            tool_res = run_web_search(queries, num_results=5)
            traj.tool_outputs.append(json.dumps(tool_res)[:4000])
            # collect top urls to fetch/process
            top_urls = []
            for block in tool_res:
                for r in block.get("results", [])[:2]:
                    if r.get("link"):
                        top_urls.append(r["link"])
            if top_urls:
                processed_raw = fetch_and_process_urls(top_urls, do_embed=False)
                # processed_raw is list of {"url":u, "processed":p}
                for p in processed_raw:
                    doc = p.get("processed", {})
                    processed.append({"url": p.get("url"), "raw_text_len": len(doc.get("raw_text","")), "short": (doc.get("raw_text","")[:800])})
                traj.tool_outputs.append(json.dumps([{"url":x["url"], "len": x["raw_text_len"]} for x in processed])[:4000])
        except Exception as e:
            # don't fail the pipeline if search or fetch fails
            print(f"[REFLEXION GRAPH] search/fetch error: {e}")

    # 3) evaluator
    evaluation = evaluator_evaluate(actor_out, env_feedback or (json.dumps(tool_res)[:2000] if tool_res else None))
    traj.evaluation = evaluation

    # 4) reflector
    reflection_json = reflector_make_reflection(actor_out, evaluation)
    try:
        from agent.reflection import Reflection
        ref = Reflection(**reflection_json) if isinstance(reflection_json, dict) else Reflection(suggestion=str(reflection_json))
    except Exception:
        # fallback to simple wrapper
        ref = type("R", (), {"missing":"", "superfluous":"", "suggestion": str(reflection_json)})
    traj.reflection = ref
    # store reflection
    try:
        append_reflection(ref)
    except Exception:
        pass

    # 5) Final revision prompt (use evidence if available)
    evidence_text = ""
    if processed:
        evidence_text = "\n".join([f"{p['url']}: (len={p['raw_text_len']}) {p['short'][:300]}" for p in processed])
    elif tool_res:
        # use snippets from search results if processed empty
        snippets = []
        for block in tool_res:
            for r in block.get("results", [])[:3]:
                t = r.get("title","") + ": " + (r.get("snippet") or "")
                snippets.append(t)
        evidence_text = "\n".join(snippets[:10])

    revision_prompt = [
        {"role":"system","content":"당신은 최종 답변을 간결히 만드는 편집자입니다. 아래 정보를 반영해 200자 이내로 정리해 주세요."},
        {"role":"user","content": f"원문질문: {command}\n\n초기답변:\n{actor_out}\n\n평가:\n{evaluation}\n\n증거(요약):\n{evidence_text}\n\n반성/개선안:\n{getattr(ref,'suggestion', '')}"}
    ]
    final = llm_chat(revision_prompt, model="gpt-4o-mini", max_tokens=400)

    evidence = {
        "queries": queries,
        "search_results": tool_res,
        "processed_docs": processed
    }

    return final, traj, evidence
