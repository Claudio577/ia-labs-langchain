from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL

def criar_agente_compliance():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.0
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscarRiscos",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Localiza indícios de riscos e não conformidades."
        ),
        Tool(
            name="ResumoCompliance",
            func=lambda texto: chain_resumo.run(texto),
            description="Cria resumo com foco em riscos e compliance."
        )
    ]

    system_prompt = """
    Você é um Assistente de Compliance e Riscos.
    Objetivo:
    - Identificar riscos
    - Mapear não conformidades
    - Avaliar governança
    - Sugerir pontos críticos

    Importante:
    - Nunca inventar alertas sem base documental
    - Ser claro e técnico
    """

    agent = create_react_agent(llm=llm, tools=tools, system_message=system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor
