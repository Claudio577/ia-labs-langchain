from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from ingest.vector_store import carregar_vector_store
from chains.summarizer import chain_resumo
from config import OPENAI_MODEL


def criar_agente_corporativo():

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )

    vectordb = carregar_vector_store()
    retriever = vectordb.as_retriever()

    tools = {
        "buscar_documentos": Tool(
            name="BuscarDocumentos",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Busca trechos relevantes nos documentos enviados."
        ),
        "resumo_executivo": Tool(
            name="ResumoExecutivo",
            func=lambda texto: chain_resumo.run(texto),
            description="Gera um resumo executivo profissional."
        )
    }

    prompt = ChatPromptTemplate.from_template("""
    Você é o assistente corporativo IA-Labs.
    Escolha uma das funções:
    - buscar_documentos
    - resumo_executivo

    Usuário disse: {query}

    Responda apenas com o resultado final.
    """)

    def executar(query):
        return llm.invoke(prompt.format(query=query)).content

    # Cria um executável estilo agente
    executor = RunnableLambda(lambda x: executar(x["input"]))

    class Wrapper:
        def run(self, texto):
            return executor.invoke({"input": texto})

    return Wrapper()
