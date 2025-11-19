from config import get_llm

def chain_resumo(texto):
    llm = get_llm()

    prompt = f"""
Resuma de forma objetiva e corporativa o conteúdo abaixo:

{texto}

Resumo:
"""

    resposta = llm.invoke(prompt)
    return resposta.content
