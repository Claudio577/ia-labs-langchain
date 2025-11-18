from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import OPENAI_MODEL

prompt = ChatPromptTemplate.from_template("""
Gere um **Resumo Executivo Profissional** para a IA-Labs.
Inclua:
- Pontos-chave
- Riscos
- Oportunidades
- Sugestões de automação com IA

Texto base:
{texto}
""")

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

def chain_resumo(texto):
    return llm.invoke(prompt.format(texto=texto)).content
