from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

from langchain.agents import AgentExecutor
from langchain.agents.react import create_react_agent_via_langgraph


def criar_agente_corporativo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo de Conteúdo",
            func=chain_resumo,
            description="Gera resumos corporativos profissionais."
        )
    ]

    # Agente REACT moderno via LangGraph (compatível!)
    agent = create_react_agent_via_langgraph(
        llm=llm,
        tools=tools,
        state_modifier="Você é um assistente corporativo profissional. Responda com clareza."
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    return executor

