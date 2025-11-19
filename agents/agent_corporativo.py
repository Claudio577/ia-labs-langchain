from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor
from langchain.agents.react.base import create_react_agent

from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL


def criar_agente_corporativo():
    """Agente corporativo compatível com LangChain moderno."""

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscarDocumentos",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Busca trechos relevantes nos documentos empresariais."
        ),
        Tool(
            name="ResumoExecutivo",
            func=lambda texto: chain_resumo.run(texto),
            description="Gera resumo executivo corporativo."
        ),
    ]

    system_prompt = """
    Você é um assistente corporativo especializado em análise de documentos,
    padronização, insights estratégicos e apoio à tomada de decisão.
    Fale de forma clara, objetiva, profissional.
    """

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        system_message=system_prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return executor

