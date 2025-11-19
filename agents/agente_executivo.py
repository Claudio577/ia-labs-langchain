from langchain.agents import AgentExecutor
from langchain.agents.react import create_react_agent
from config import get_llm

def criar_agente_executivo():
    llm = get_llm()

    prompt = """
Você é um consultor executivo sênior. 
Sua missão: orientar estratégias, decisões e visões corporativas.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False)
