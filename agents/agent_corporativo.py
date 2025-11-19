from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_MODEL


# ====== Ferramenta exemplo — buscar em documentos RAG ======
@tool
def buscar_documentos(query: str) -> str:
    """
    Ferramenta que simula busca em base de conhecimento.
    Substitua pelo seu vector_store real depois.
    """
    return f"Buscando informações relacionadas a: {query}"


tools = [buscar_documentos]


# ====== PROMPT DO AGENTE ======
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um agente corporativo especializado em análise de documentos "
     "e respostas empresariais. Use ferramentas sempre que necessário."),
    ("human", "{input}")
])


def criar_agente_corporativo():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    agent = create_openai_tools_agent(
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

