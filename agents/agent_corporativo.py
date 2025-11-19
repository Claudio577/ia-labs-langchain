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
            description="Gera resumos objetivos para documentos longos ou relatórios."
        )
    ]

    prompt = """
Você é um assistente corporativo especializado.
Forneça respostas claras, profissionais e diretas.
Se precisar resumir conteúdo, use a ferramenta de resumo.
"""

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=False)

