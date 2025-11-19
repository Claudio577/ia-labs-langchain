from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

def criar_agente_financeiro():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Financeiro",
            func=chain_resumo,
            description="Cria resumos técnicos e análises financeiras objetivas."
        )
    ]

    prompt = """
Você é um consultor financeiro profissional.
Forneça análises claras, objetivas e baseadas em dados.
Evite recomendações diretas de investimento.
Use a ferramenta de resumo quando necessário.
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
