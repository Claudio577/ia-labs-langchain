from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_MODEL


# ======== Ferramenta de Busca RAG (placeholder) ========
@tool
def buscar_documentos(query: str) -> str:
    """Busca conteúdos relevantes em documentos enviados."""
    return f"Buscando informações sobre: {query}"


tools = [buscar_documentos]


# ======== PROMPT (sistema) ========
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente corporativo especializado em análise documental, "
     "resumos e respostas profissionais. Sempre utilize ferramentas quando necessário."),
    ("human", "{input}")
])


def criar_agente_corporativo():
    # Modelo OpenAI
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
    )

    # Agente baseado em Funções OpenAI (Functions)
    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Executor final
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False
    )

    return executor

