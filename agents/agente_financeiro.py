from langchain.agents import AgentExecutor
from langchain.agents.react import create_react_agent
from config import get_llm

def criar_agente_financeiro():
    llm = get_llm()

    prompt = """
Você é um analista financeiro corporativo. 
Explique indicadores e análises econômicas de forma clara e objetiva.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False)
