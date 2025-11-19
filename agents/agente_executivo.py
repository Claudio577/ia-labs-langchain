from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain_core.prompts import PromptTemplate

def criar_agente_executivo():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Executivo",
            func=chain_resumo,
            description="Gera resumos executivos estratégicos e diretos."
        )
    ]

    prompt = PromptTemplate(
        template="""
Você é um consultor executivo especializado em estratégia, gestão e liderança.
Responda com visão estratégica, objetividade e foco em impacto.

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
