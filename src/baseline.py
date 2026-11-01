import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llm import get_llm

Q_PREFIX = "Represent this sentence for searching relevant passages: "
PROMPT = """You answer questions about India's DPDP Act, 2023, using ONLY the context below.
Cite sections like [Section 9]. If the context does not contain the answer, reply exactly: Not found in the Act.
Do not give legal advice.

Context:
{context}

Question: {question}
Answer:"""

def answer(question, k=4):
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                encode_kwargs={"normalize_embeddings": True})
    db = Chroma(persist_directory="chroma_db", embedding_function=emb, collection_name="dpdp")
    docs = db.similarity_search(Q_PREFIX + question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    reply = get_llm().invoke(PROMPT.format(context=context, question=question)).content
    return reply, docs

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Can a company process a child's data without parental consent?"
    reply, docs = answer(q)
    print(reply)
    print("\nRetrieved:", [f"S{d.metadata['section']}" for d in docs])