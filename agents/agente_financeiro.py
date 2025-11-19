from langchain.agents import AgentExecutor, create_react_agent
from config import get_llm

def criar_agente_financeiro():
    llm = get_llm()

    prompt = """
Você é um analista financeiro corporativo.
Gere análises, interpretações de indicadores e explicações de relatórios financeiros.
Use linguagem simples e profissional.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)

    return AgentExecutor(agent=agent, tools=[], verbose=False)
