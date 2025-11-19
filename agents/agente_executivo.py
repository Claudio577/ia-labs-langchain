from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL


def criar_agente_executivo():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscaEstratégica",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Busca insights estratégicos e informações relevantes."
        ),
        Tool(
            name="ResumoC-Level",
            func=lambda texto: chain_resumo.run(texto),
            description="Resumo direto ao ponto para diretores e executivos."
        )
    ]

    agent = create_react_agent(llm=llm, tools=tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor

