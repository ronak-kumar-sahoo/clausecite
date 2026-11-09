import re, sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from llm import get_llm
from retrieval import search, _db, P

class State(TypedDict, total=False):
    question: str
    in_scope: bool
    docs: List
    answer: str
    attempts: int
    verified: bool
    query: str
    retries: int
    relevant: bool
    faithful: bool    

SCOPE = """You route questions for an assistant that only answers about India's Digital Personal Data Protection (DPDP) Act, 2023: personal data, consent, Data Fiduciaries, Data Principals, penalties, the Data Protection Board, appeals.
Reply with exactly one word: IN if the question is about that Act or personal data protection law, otherwise OUT.
Question: {question}"""

GEN = """Answer using ONLY the context below, which is from India's DPDP Act, 2023.
Cite only sections whose headers appear in the context. Put every citation in square brackets, like [Section 9] or [Section 6, Section 8], and use [Schedule] for the penalty table. Always end your answer with its citations. Never cite a section that is not in the context. If the context does not contain the answer, reply exactly: Not found in the Act.

Context:
{context}

Question: {question}
Answer:"""

def classify(s):
    dist = _db.similarity_search_with_score(P + s["question"], k=1)[0][1]
    return {"in_scope": dist < 0.85, "attempts": 0, "retries": 0}

GRADE = """Do these excerpts from India's DPDP Act, 2023 contain enough information to answer the question? Reply with exactly one word: YES or NO.
Excerpts:
{context}
Question: {question}"""

REWRITE = """Rewrite the question as a short search query using the vocabulary of India's DPDP Act, 2023 (for example Data Fiduciary, Data Principal, consent, personal data breach, Board, penalty). Reply with the query only.
Question: {question}"""

def retrieve(s):
    return {"docs": search(s.get("query") or s["question"], k=4)}

def grade(s):
    ctx = "\n\n".join(d.page_content for d in s["docs"])    
    r = get_llm().invoke(GRADE.format(context=ctx, question=s["question"])).content.strip().upper()
    return {"relevant": r.startswith("YES")}

def rewrite(s):
    q = get_llm().invoke(REWRITE.format(question=s["question"])).content.strip()
    return {"query": q, "retries": s.get("retries", 0) + 1}

def not_found(s):
    return {"answer": "Not found in the Act."}

def generate(s):
    ctx = "\n\n".join(d.page_content for d in s["docs"])
    a = get_llm().invoke(GEN.format(context=ctx, question=s["question"])).content.strip()
    return {"answer": a, "attempts": s["attempts"] + 1}

def citations(a):
    out = set()
    for b in re.findall(r"[\[\u3010]([^\]\u3011]*)[\]\u3011]", a):
        b = re.sub(r"\(.*?\)", "", b)
        if re.search(r"[Ss]ections?\b", b):
            out |= set(re.findall(r"\b(\d{1,2})\b", b))
        if "Schedule" in b:
            out.add("Schedule")
    return out

def verify(s):
    a = s["answer"]
    if a.lower().startswith("not found"):
        return {"verified": True}
    cited = citations(a)   
    have = {d.metadata["section"] for d in s["docs"]}
    ok = bool(cited) and cited <= have
    out = {"verified": ok}
    if not ok and s["attempts"] >= 2:
        out["answer"] = "Not found in the Act (no verified, cited answer could be produced)."
    return out

def refuse(s):
    return {"answer": "I can only answer questions about the DPDP Act, 2023."}

FAITH = """Excerpts from India's DPDP Act, 2023:
{context}

Answer to check:
{answer}

Reply NO only if the answer states something that contradicts the excerpts or is not in them. Summaries and paraphrases of the excerpts are fine. Reply with exactly one word: YES or NO."""

def faithful(s):
    if s["answer"].lower().startswith("not found"):
        return {"faithful": True}
    ctx = "\n\n".join(d.page_content for d in s["docs"])
    r = get_llm().invoke(FAITH.format(context=ctx, answer=s["answer"])).content.strip().upper()
    ok = r.startswith("YES")
    out = {"faithful": ok}
    if not ok and s["attempts"] >= 2:
        out["answer"] = "Not found in the Act (no fully supported answer could be produced)."
    return out

route_scope = lambda s: "retrieve" if s["in_scope"] else "refuse"
route_verify = lambda s: "faithful" if s["verified"] else (END if s["attempts"] >= 2 else "generate")
route_faith = lambda s: END if s["faithful"] or s["attempts"] >= 2 else "generate"
route_grade = lambda s: "generate" if s["relevant"] else ("rewrite" if s["retries"] < 2 else "not_found")

g = StateGraph(State)
for name, fn in [("classify", classify), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                 ("generate", generate), ("verify", verify),("faithful", faithful), ("refuse", refuse), ("not_found", not_found)]:
    g.add_node(name, fn)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", route_scope, {"retrieve": "retrieve", "refuse": "refuse"})
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route_grade, {"generate": "generate", "rewrite": "rewrite", "not_found": "not_found"})
g.add_edge("rewrite", "retrieve")
g.add_edge("generate", "verify")
g.add_conditional_edges("verify", route_verify, {"faithful": "faithful", END: END, "generate": "generate"})
g.add_conditional_edges("faithful", route_faith, {END: END, "generate": "generate"})
g.add_edge("refuse", END)
g.add_edge("not_found", END)
app = g.compile()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Can a company process a child's data without parental consent?"
    out = app.invoke({"question": q})
    print(out["answer"])
    print("\nRetrieved:", [f"S{d.metadata['section']}" for d in out.get("docs", [])],
          "| query:", out.get("query"), "| retries:", out.get("retries"),
          "| attempts:", out.get("attempts"), "| verified:", out.get("verified"))    