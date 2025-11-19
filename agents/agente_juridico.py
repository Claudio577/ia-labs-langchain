from langchain.agents import AgentExecutor, create_react_agent
from config import get_llm

def criar_agente_juridico():
    llm = get_llm()

    prompt = """
Você é um analista jurídico. 
Pode interpretar contratos, cláusulas e oferecer explicações jurídicas simples.
Não forneça aconselhamento legal oficial. Apenas interpretações.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)

    return AgentExecutor(agent=agent, tools=[], verbose=False)

