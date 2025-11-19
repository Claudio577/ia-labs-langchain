from langchain.agents import AgentExecutor
from langchain.agents.react import create_react_agent
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

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=False)

