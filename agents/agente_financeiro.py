from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL

def criar_agente_financeiro():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscarDadosFinanceiros",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Busca informações financeiras nos documentos."
        ),
        Tool(
            name="ResumoFinanceiro",
            func=lambda texto: chain_resumo.run(texto),
            description="Cria resumo com foco financeiro."
        )
    ]

    system_prompt = """
    Você é um Analista Financeiro IA especializado em:
    - DRE, fluxo de caixa, balanço patrimonial
    - Análises de risco
    - Indicadores financeiros

    Regras:
    - Seja objetivo
    - Não invente números
    - Só use dados presentes nos documentos enviados
    """

    agent = create_react_agent(llm=llm, tools=tools, system_message=system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor
