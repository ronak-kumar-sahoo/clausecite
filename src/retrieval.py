import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

P = "Represent this sentence for searching relevant passages: "
_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                             encode_kwargs={"normalize_embeddings": True})
_db = Chroma(persist_directory="chroma_db", embedding_function=_emb, collection_name="dpdp")
_all = _db.get()
_docs = list(zip(_all["documents"], _all["metadatas"]))
_key = lambda m: (m["section"], m["part"])
_by_key = {_key(m): Document(page_content=t, metadata=m) for t, m in _docs}
_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
_bm25 = BM25Okapi([_tok(t) for t, _ in _docs])
_rerank = None

def search(q, k=4, mode="hybrid_rerank", pool=20):
    global _rerank
    vec = [_key(d.metadata) for d in _db.similarity_search(P + q, k=pool)]
    if mode == "vector":
        return [_by_key[x] for x in vec[:k]]
    sc = _bm25.get_scores(_tok(q))
    bm = [_key(_docs[i][1]) for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:pool]]
    fused = {}
    for ranking in (vec, bm):
        for r, key in enumerate(ranking):
            fused[key] = fused.get(key, 0) + 1 / (60 + r)
    keys = sorted(fused, key=fused.get, reverse=True)
    if mode == "hybrid":
        return [_by_key[x] for x in keys[:k]]
    if _rerank is None:
        _rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cand = keys[:pool]
    scores = _rerank.predict([(q, _by_key[x].page_content) for x in cand])
    ranked = [x for _, x in sorted(zip(scores, cand), key=lambda z: -z[0])]
    best, count = [], {}
    for x in ranked:
        if count.get(x[0], 0) < 2:
            best.append(x)
            count[x[0]] = count.get(x[0], 0) + 1
        if len(best) == k:
            break
    return [_by_key[x] for x in best]    