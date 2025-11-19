from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

from config import get_llm
from ingest.vector_store import get_vector_store  # você precisa ter isso

# -------------------------------------------------
# FERRAMENTA RAG – consulta documentos enviados
# -------------------------------------------------
def buscar_documentos(query: str):
    """
    Consulta a base vetorial e retorna textos relevantes
    para apoiar decisões executivas.
    """
    try:
        vectorstore = get_vector_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        if not docs:
            return "Nenhum documento relevante encontrado."

        texto = "\n\n".join(d.page_content for d in docs)
        return texto

    except Exception as e:
        return f"Erro ao buscar documentos: {e}"


# -------------------------------------------------
# PROMPT EXECUTIVO PROFISSIONAL
# -------------------------------------------------
system_prompt = """
Você é um **Executivo Sênior de alto nível**, com mais de 20 anos de experiência 
em estratégia corporativa, gestão de riscos, inovação, eficiência operacional e 
tomada de decisão baseada em dados.

Seu estilo deve ser:
- Direto e objetivo  
- Baseado em impacto e clareza  
- Enxuto, estruturado e estratégico  
- Focado em riscos, oportunidades e decisões  
- Sempre profissional e alinhado ao contexto de negócios  

Quando usar informações de documentos (RAG), cite insights sem copiar demais.
Quando não houver dados suficientes, seja transparente.
"""

# -------------------------------------------------
# CRIAÇÃO DO AGENTE EXECUTIVO
# -------------------------------------------------
def criar_agente_executivo():
    llm = get_llm()

    tools = [
        Tool(
            name="Pesquisa Documentos Corporativos",
            func=buscar_documentos,
            description=(
                "Busca informações relevantes nos PDFs, contratos, políticas "
                "e documentos enviados pelo usuário para auxiliar na tomada "
                "de decisão executiva."
            )
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        system_message=system_prompt,
    )

    return agent
