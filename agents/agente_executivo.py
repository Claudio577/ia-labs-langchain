from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

def criar_agente_executivo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Executivo",
            func=chain_resumo,
            description="Gera resumos curtos e objetivos para tomada de decisão."
        )
    ]

    prompt = """
Você é um assistente executivo especializado em tomada de decisão.
Produza respostas extremamente objetivas e práticas.
Quando necessário, utilize as ferramentas disponíveis.
"""

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    return executor
