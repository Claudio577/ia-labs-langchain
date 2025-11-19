import os
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",  # MODELO COMPATÍVEL
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
