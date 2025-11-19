from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from config import get_llm

def criar_agente_compliance():
    llm = get_llm()

    prompt = """
Você é um especialista em compliance.
Explique políticas internas, riscos e boas práticas de governança.
Mantenha as respostas formais, claras e objetivas.
"""

    agent = create_react_agent(llm=llm, tools=[], prompt=prompt)

    return AgentExecutor(agent=agent, tools=[], verbose=False)


