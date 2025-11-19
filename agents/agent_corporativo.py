from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_MODEL


# ======== Ferramenta de busca ========
@tool
def buscar_documentos(query: str) -> str:
    """Busca informações relevantes nos documentos enviados."""
    return f"Buscando informações nos documentos sobre: {query}"


tools = [buscar_documentos]


# ======== Prompt do agente ========
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente corporativo altamente profissional da IA-Labs. "
     "Analise dados, responda dúvidas e gere insights."),
    ("human", "{input}")
])


def criar_agente_corporativo():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

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


