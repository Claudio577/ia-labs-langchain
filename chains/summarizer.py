from config import get_llm

def chain_resumo(texto):
    llm = get_llm()

    prompt = f"""
Resuma o conteúdo abaixo de forma corporativa, objetiva e clara:

{texto}

Resumo:
"""

    resposta = llm.invoke(prompt)
    return resposta.content
