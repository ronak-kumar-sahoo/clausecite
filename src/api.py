import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from graph import app as graph, citations

api = FastAPI(title="ClauseCite")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/ask")
def ask(q: Q):
    question = q.question.strip()[:500]
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    try:
        out = graph.invoke({"question": question})
    except Exception:
        raise HTTPException(status_code=503, detail="The language model is busy or unavailable. Please try again shortly.")
    answer = out.get("answer", "")
    cited, seen, sources = citations(answer), set(), []
    for d in out.get("docs", []):
        m = d.metadata
        if m["section"] in cited and m["section"] not in seen:
            seen.add(m["section"])
            sources.append({"section": m["section"], "title": m["title"], "page": m["page"]})
    return {"answer": answer, "sources": sources, "retries": out.get("retries", 0),
            "verified": out.get("verified"), "disclaimer": "Informational only, not legal advice."}

web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
api.mount("/", StaticFiles(directory=web_dir, html=True), name="web")