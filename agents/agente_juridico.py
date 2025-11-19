from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL


def criar_agente_juridico():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.0  # jurídico sempre mais conservador
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscaLegal",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Busca cláusulas e informações jurídicas nos documentos."
        ),
        Tool(
            name="ResumoJuridico",
            func=lambda texto: chain_resumo.run(texto),
            description="Resumo jurídico detalhado, sem perder precisão."
        )
    ]

    agent = create_react_agent(llm=llm, tools=tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor
