from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from chains.summarizer import chain_resumo
from config import get_llm
from langchain_core.prompts import PromptTemplate

def criar_agente_compliance():
    llm = get_llm()

    tools = [
        Tool(
            name="Resumo Compliance",
            func=chain_resumo,
            description="Resume documentos e normas de compliance."
        )
    ]

    prompt = PromptTemplate(
        template="""
Você é especialista em compliance, ética corporativa e governança.
Responda com foco em políticas internas, boas práticas e riscos.

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

