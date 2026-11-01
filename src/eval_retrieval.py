import json
from retrieval import search

K = 4
rows = [json.loads(l) for l in open("eval/gold.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if r["sections"]]
for mode in ("vector", "hybrid", "hybrid_rerank"):
    hit = full = 0
    for r in rows:
        got = [d.metadata["section"] for d in search(r["q"], K, mode)]
        want = set(r["sections"])
        hit += bool(want & set(got))
        ok = want <= set(got)
        full += ok
        if mode == "hybrid_rerank" and not ok:
            print("MISS |", r["q"], "| want", sorted(want), "| got", got)
    print(f"{mode:14} hit@{K}: {hit}/{len(rows)}  full-recall@{K}: {full}/{len(rows)}")