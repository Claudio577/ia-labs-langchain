from langchain.agents import AgentExecutor
from langchain.agents.react import create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
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

    # Carrega o prompt correto do LangChain Hub
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )
