import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

K = 4
P = "Represent this sentence for searching relevant passages: "
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                            encode_kwargs={"normalize_embeddings": True})
db = Chroma(persist_directory="chroma_db", embedding_function=emb, collection_name="dpdp")

rows = [json.loads(l) for l in open("eval/gold.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if r["sections"]]
hit = full = 0
for r in rows:
    got = [d.metadata["section"] for d in db.similarity_search(P + r["q"], k=K)]
    want = set(r["sections"])
    h, f = bool(want & set(got)), want <= set(got)
    hit += h
    full += f
    if not f:
        print("MISS" if not h else "PARTIAL", "|", r["q"], "| want", sorted(want), "| got", got)
print(f"hit@{K}: {hit}/{len(rows)}  full-recall@{K}: {full}/{len(rows)}")