from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_community.agent_toolkits import initialize_agent
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

    agente = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agente

