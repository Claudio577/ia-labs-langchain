from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain_core.prompts import PromptTemplate

def criar_agente_financeiro():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Financeiro",
            func=chain_resumo,
            description="Cria resumos com foco em finanças, indicadores e análises."
        )
    ]

    prompt = PromptTemplate(
        template="""
Você é um analista financeiro corporativo especialista em dados, relatórios e indicadores.
Foque em clareza, precisão e análise objetiva.

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

