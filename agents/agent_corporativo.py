from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL

def criar_agente_corporativo():

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
            description="Busca trechos relevantes nos documentos enviados."
        ),
        Tool(
            name="ResumoExecutivo",
            func=lambda texto: chain_resumo.run(texto),
            description="Gera resumo executivo profissional."
        )
    ]

    # Cria o agente no estilo ReAct (novo padrão LangChain)
    agent = create_react_agent(llm=llm, tools=tools)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return executor

