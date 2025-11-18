from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from chains.summarizer import chain_resumo
from ingest.vector_store import carregar_vector_store
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
            name="Buscar em Documentos IA-Labs",
            func=retriever.get_relevant_documents,
            description="Usado para buscar informações nos documentos enviados."
        ),
        Tool(
            name="Gerar Resumo Executivo",
            func=chain_resumo.run,
            description="Cria um resumo executivo claro e corporativo."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent
