from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

def criar_agente_compliance():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo de Compliance",
            func=chain_resumo,
            description="Produz resumos formais e análises para auditorias e conformidade."
        )
    ]

    prompt = """
Você é um especialista em compliance corporativo.
Forneça respostas formais, regulamentadas, sem opinião pessoal.
Use normas, políticas, processos e boas práticas.
Sempre mantenha tom profissional e técnico.
Se necessário, utilize ferramentas de resumo.
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
