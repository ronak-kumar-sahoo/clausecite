import json, re, shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

data = json.load(open("data/processed/sections.json", encoding="utf-8"))
MAX = 1000
texts, metas, ids = [], [], []

def pack(text):
    blocks = re.split(r"\n(?=\([a-z0-9]{1,4}\) )", text)
    chunks, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) > MAX:
            chunks.append(cur)
            cur = b
        else:
            cur = (cur + "\n" + b) if cur else b
    chunks.append(cur)
    return chunks

for s in data["sections"]:
    head = f"DPDP Act 2023, Section {s['section']}. {s['title']} (Chapter {s['chapter']}: {s['chapter_title'].title()})"
    for i, c in enumerate(pack(s["text"]), 1):
        texts.append(f"{head}\n{c}")
        metas.append({"section": str(s["section"]), "title": s["title"],
                      "chapter": s["chapter"], "page": s["page"], "part": i})
        ids.append(f"s{s['section']}-{i}")

texts.append("DPDP Act 2023, The Schedule (penalties)\n" + data["schedule"])
metas.append({"section": "Schedule", "title": "The Schedule (penalties)", "chapter": "", "page": 24, "part": 1})
ids.append("schedule-1")

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                            encode_kwargs={"normalize_embeddings": True})
shutil.rmtree("chroma_db", ignore_errors=True)
db = Chroma.from_texts(texts, emb, metadatas=metas, ids=ids,
                       persist_directory="chroma_db", collection_name="dpdp")
print("chunks indexed:", len(texts))

q = "Represent this sentence for searching relevant passages: Can a company process a child's data without parental consent?"
for d in db.similarity_search(q, k=3):
    print("->", "Section", d.metadata["section"], d.metadata["title"], "| p", d.metadata["page"])