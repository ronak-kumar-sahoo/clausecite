import json, os, re, sys, time
from graph import app

rows = [json.loads(l) for l in open("eval/gold.jsonl", encoding="utf-8") if l.strip()]
step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rows = rows[::step]
res, t0, errors = [], time.time(), 0
for r in rows:
    time.sleep(3)
    try:
        a = app.invoke({"question": r["q"]}).get("answer", "")    
    except Exception as e:
        a = f"ERROR: {e}"
    if a.startswith("ERROR"):
        print("ERROR (not scored) |", r["q"])
        errors += 1
        continue        
    low = a.lower()
    refused, notfound = low.startswith("i can only answer"), low.startswith("not found")
    cited = set(re.findall(r"[Ss]ection\s+(\d{1,2})", a))
    if re.search(r"schedule", a, re.I):
        cited.add("Schedule")
    want, t = set(r["sections"]), r["type"]
    if t == "offtopic":
        ok = refused
    elif t == "unanswerable":
        ok = notfound or refused
    else:
        ok = (not refused) and (not notfound) and bool(want & cited)
    res.append({"q": r["q"], "type": t, "ok": ok, "answer": a})
    if not ok:
        print("FAIL |", t, "|", r["q"], "| want", sorted(want), "| cited", sorted(cited), "|", a[:90].replace("\n", " "))

for t in ("definition", "lookup", "multi", "unanswerable", "offtopic"):
    g = [x for x in res if x["type"] == t]
    if g:
        print(f"{t:13} {sum(x['ok'] for x in g)}/{len(g)}")
model = os.getenv("OLLAMA_MODEL", "llama3.2") if os.getenv("LLM_PROVIDER") == "ollama" else os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
print("model:", model, "| minutes:", round((time.time() - t0) / 60, 1))
json.dump(res, open("eval/graph_results.json", "w", encoding="utf-8"), indent=2)
print("unscored errors:", errors)