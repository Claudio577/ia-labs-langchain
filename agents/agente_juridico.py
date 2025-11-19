from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL

def criar_agente_juridico():
    """Agente jurídico para análise de contratos e documentos legais."""

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.0  # jurídico exige precisão
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = [
        Tool(
            name="BuscarClausulas",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Localiza cláusulas e seções relevantes."
        ),
        Tool(
            name="ResumoLegal",
            func=lambda texto: chain_resumo.run(texto),
            description="Gera resumo jurídico objetivo."
        )
    ]

    system_prompt = """
    Você é um Assistente Jurídico especializado em:
    - Análise de contratos
    - Compliance
    - Riscos legais
    - Due diligence

    Boas práticas:
    - Não ofereça aconselhamento jurídico final
    - Não invente leis
    - Baseie respostas APENAS nos documentos enviados
    """

    agent = create_react_agent(llm=llm, tools=tools, system_message=system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor
