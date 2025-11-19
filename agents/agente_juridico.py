from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain.agents import initialize_agent, AgentType

def criar_agente_juridico():
    llm = get_llm()

    tools = [
        Tool(
            name="Análise Jurídica",
            func=chain_resumo,
            description="Resume e simplifica documentos jurídicos."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )

    return agent
