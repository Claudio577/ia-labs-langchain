from langchain.agents import AgentExecutor
from langchain.agents.react import create_react_agent
from config import get_llm

def criar_agente_juridico():
    llm = get_llm()

    prompt = """
Você é um analista jurídico especializado. 
Explique cláusulas, documentos e termos jurídicos de forma simples.
Não forneça aconselhamento legal.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False)
