from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ingest.vector_store import carregar_vector_store
from config import OPENAI_MODEL

def criar_agente_executivo():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Você é o **Agente Executivo IA-Labs**, especialista em:
    - Resumos executivos
    - Insights estratégicos
    - Identificação de riscos e oportunidades
    - Plano de ação
    - KPIs
    - Visão consultiva corporativa

    Documentos relevantes:
    {contexto}

    Pergunta:
    {input}

    Gere uma resposta estruturada:

    🎯 Insight principal  
    ⚠️ Riscos identificados  
    💡 Oportunidades observadas  
    📈 KPIs recomendados  
    🧠 Ações sugeridas pela IA-Labs  
    """)

    def executar(texto):
        docs = retriever.get_relevant_documents(texto)
        contexto = "\n\n".join([d.page_content for d in docs])

        return llm.invoke(prompt.format(
            contexto=contexto,
            input=texto
        )).content

    class Wrapper:
        def run(self, texto):
            return executar(texto)

    return Wrapper()
