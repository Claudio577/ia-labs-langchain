from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain_core.prompts import PromptTemplate

def criar_agente_corporativo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo de Conteúdo",
            func=chain_resumo,
            description="Gera resumos corporativos profissionais."
        )
    ]

    prompt = PromptTemplate(
        template="""
Você é um assistente corporativo especializado em comunicação empresarial.
Responda com clareza, objetividade e foco em tomada de decisão.

Pergunta do usuário: {input}
""",
        input_variables=["input"],
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=False)
