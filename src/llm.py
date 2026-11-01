import os
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    if os.getenv("LLM_PROVIDER", "groq") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
    from langchain_groq import ChatGroq
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)