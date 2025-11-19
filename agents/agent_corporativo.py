from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from config import OPENAI_MODEL

# ======== Ferramenta RAG (placeholder) ========
@tool
def buscar_documentos(query: str) -> str:
    """Busca informações relevantes nos documentos enviados."""
    return f"Buscando informações sobre: {query}"

tools = [buscar_documentos]

# ======== Prompt ========
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente corporativo profissional da IA-Labs."
    ),
    ("user", "{input}")
])

def criar_agente_corporativo():

    llm = ChatOpenAI(
        temperature=0.2,
        model=OPENAI_MODEL
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
