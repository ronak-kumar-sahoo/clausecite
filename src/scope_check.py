import json
from retrieval import _db, P

rows = [json.loads(l) for l in open("eval/gold.jsonl", encoding="utf-8") if l.strip()]
for t in ("definition", "lookup", "multi", "unanswerable", "offtopic"):
    s = [round(_db.similarity_search_with_score(P + r["q"], k=1)[0][1], 2) for r in rows if r["type"] == t]
    print(f"{t:13} min {min(s)}  max {max(s)}")