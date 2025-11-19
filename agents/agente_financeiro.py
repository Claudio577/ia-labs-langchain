from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ingest.vector_store import carregar_vector_store
from config import OPENAI_MODEL

def criar_agente_financeiro():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Você é o **Agente Financeiro IA-Labs**, especialista em:
    - análise financeira
    - riscos
    - oportunidades
    - tendências
    - KPIs financeiros
    - fluxo de caixa
    - insights quantitativos

    Documentos relevantes:
    {contexto}

    Pergunta:
    {input}

    Gere uma resposta profissional contendo:

    💰 Insight financeiro  
    ⚠️ Riscos financeiros  
    📈 Oportunidades  
    📊 Indicadores e KPIs  
    🧮 Análise numérica (se houver dados)  
    🧠 Ações recomendadas  
    """)

    def executar(texto):
        docs = retriever.get_relevant_documents(texto)
        contexto = "\n\n".join([d.page_content for d in docs])
        return llm.invoke(prompt.format(contexto=contexto, input=texto)).content

    class Wrapper:
        def run(self, texto):
            return executar(texto)

    return Wrapper()
