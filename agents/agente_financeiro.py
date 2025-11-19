from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL


def criar_agente_financeiro():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.15
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscarIndicadores",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Busca dados financeiros, projeções e KPIs."
        ),
        Tool(
            name="ResumoFinanceiro",
            func=lambda texto: chain_resumo.run(texto),
            description="Cria resumo financeiro com métricas importantes."
        )
    ]

    agent = create_react_agent(llm=llm, tools=tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor
