from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain_core.prompts import PromptTemplate

def criar_agente_juridico():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Jurídico",
            func=chain_resumo,
            description="Simplifica e resume conteúdos jurídicos."
        )
    ]

    prompt = PromptTemplate(
        template="""
Você é um advogado corporativo especializado em compliance, contratos e legislação.
Explique com precisão, linguagem clara e sem emitir parecer jurídico formal.

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
