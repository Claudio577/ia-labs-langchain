from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ingest.vector_store import carregar_vector_store
from config import OPENAI_MODEL

def criar_agente_compliance():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Você é o **Agente de Compliance IA-Labs**, especializado em:
    - normas internas
    - políticas corporativas
    - identificação de não conformidades
    - riscos operacionais
    - governança
    - recomendações de conformidade

    Documentos relevantes:
    {contexto}

    Pergunta:
    {input}

    Gere uma resposta de compliance:

    🛡️ Pontos críticos  
    ⚠️ Não conformidades identificadas  
    📌 Riscos operacionais  
    📋 Políticas relacionadas  
    🧠 Recomendações IA-Labs  
    """)

    def executar(texto):
        docs = retriever.get_relevant_documents(texto)
        contexto = "\n\n".join([d.page_content for d in docs])
        return llm.invoke(prompt.format(contexto=contexto, input=texto)).content

    class Wrapper:
        def run(self, texto):
            return executar(texto)

    return Wrapper()
