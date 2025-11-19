from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits import initialize_react_agent


def criar_agente_corporativo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo de Conteúdo",
            func=chain_resumo,
            description="Gera resumos corporativos profissionais."
        )
    ]

    # Inicializa um agente REACT totalmente compatível
    agent = initialize_react_agent(
        tools=tools,
        llm=llm,
        verbose=False
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    return executor

