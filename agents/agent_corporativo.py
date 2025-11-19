from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm

def criar_agente_corporativo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo de Conteúdo",
            func=chain_resumo,
            description="Gera resumos corporativos profissionais."
        )
    ]

    prompt = """
Você é um assistente corporativo profissional.
Forneça respostas claras, diretas e objetivas.
Se precisar, use ferramentas de resumo ou análise.
"""

    # Cria agente REACT no novo formato do LangChain
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Executor do agente
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    return executor

