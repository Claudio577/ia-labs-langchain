from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent import (
    OpenAIFunctionsAgent,
)
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_MODEL


# ======== Ferramenta de busca (simulação) ========
@tool
def buscar_documentos(query: str) -> str:
    """Busca informações relevantes nos documentos."""
    return f"Buscando informações sobre: {query}"


tools = [buscar_documentos]


# ======== PROMPT ========
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente corporativo profissional. "
     "Analise documentos, gere resumos e responda com clareza."),
    ("human", "{input}")
])


def criar_agente_corporativo():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    # Agente baseado em OpenAI Functions (compatível com todas as versões)
    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False
    )

    return executor

