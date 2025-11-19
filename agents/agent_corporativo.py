from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

from langchain.agents import AgentExecutor
from langchain.agents.react import ReactAgent
from langchain import hub


def criar_agente_corporativo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo de Conteúdo",
            func=chain_resumo,
            description="Gera resumos corporativos profissionais."
        )
    ]

    # Prompt oficial do REACT
    prompt = hub.pull("hwchase17/react")

    agent = ReactAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    return executor
