from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import OPENAI_MODEL

prompt = PromptTemplate(
    input_variables=["texto"],
    template="""
    Gere um **Resumo Executivo Profissional** para a IA-Labs.
    Foque em:
    - Pontos-chave
    - Riscos
    - Melhorias
    - Oportunidades de automação
    - Uso de IA e dados

    Texto base:
    {texto}
    """
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
chain_resumo = LLMChain(llm=llm, prompt=prompt)
