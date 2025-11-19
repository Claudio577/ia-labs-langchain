from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import get_llm
from chains.summarizer import chain_resumo


def criar_agente_corporativo():
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """
        Você é um assistente corporativo profissional especializado em responder com clareza,
        objetividade e foco executivo.

        Caso o usuário peça resumo, chame esta função de resumo:
        {resumo_func}

        Pergunta:
        {input}
        """
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )

    return chain
