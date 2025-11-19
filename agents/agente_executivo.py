from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain.agents import initialize_agent, AgentType


def criar_agente_executivo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Executivo",
            func=chain_resumo,
            description="Gera resumos estratégicos para decisões executivas."
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
