from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

def criar_agente_juridico():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Jurídico",
            func=chain_resumo,
            description="Cria resumos jurídicos objetivos e estruturados."
        )
    ]

    prompt = """
Você é um assistente jurídico especializado.
Explique utilizando linguagem clara, objetiva e sem emitir opiniões pessoais.
Use as ferramentas quando necessário.
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
