from langchain.agents import AgentExecutor, create_react_agent
from config import get_llm

def criar_agente_executivo():
    llm = get_llm()

    prompt = """
Você é um consultor executivo sênior. 
Seu papel: apoiar decisões estratégicas, organizar ideias e propor planos de ação.
Responda de forma objetiva, clara e com foco em negócios.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)

    return AgentExecutor(agent=agent, tools=[], verbose=False)

