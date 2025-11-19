from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ingest.vector_store import carregar_vector_store
from config import OPENAI_MODEL

def criar_agente_juridico():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Você é o **Agente Jurídico IA-Labs (ContractAI)**.
    Especialidades:
    - Análise contratual
    - Identificação de riscos legais
    - Obrigações das partes
    - Cláusulas críticas
    - Recomendações jurídicas

    Documentos relevantes:
    {contexto}

    Pergunta:
    {input}

    Gere uma resposta jurídica estruturada:

    ⚖️ Cláusulas relevantes  
    ⚠️ Riscos legais detectados  
    📌 Obrigações das partes  
    🔍 Observações importantes  
    🛡️ Recomendações para mitigação  
    """)

    def executar(texto):
        docs = retriever.get_relevant_documents(texto)
        contexto = "\n\n".join([d.page_content for d in docs])

        return llm.invoke(prompt.format(contexto=contexto, input=texto)).content

    class Wrapper:
        def run(self, texto):
            return executar(texto)

    return Wrapper()
