from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableConfig
from langchain.agents.react.base import ReActSingleActionAgent
from langchain_core.prompts import ChatPromptTemplate
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
            description="Busca trechos relevantes dos documentos enviados."
        ),
        Tool(
            name="ResumoExecutivo",
            func=lambda texto: chain_resumo.run(texto),
            description="Gera um resumo executivo profissional."
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um agente corporativo da IA-Labs. "
         "Use ferramentas quando necessário e responda com clareza."),
        ("human", "{input}")
    ])

    agent = ReActSingleActionAgent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    def run(query):
        return agent.invoke(
            {"input": query},
            config=RunnableConfig()
        )["output"]

    return type("Executor", (), {"run": run})
