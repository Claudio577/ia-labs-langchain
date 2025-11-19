from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def criar_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="ia_labs_docs"
    )
# -------------------------------------------------
# Permite que os agentes acessem a vector store
# -------------------------------------------------

_vector_store_global = None

def criar_vector_store(chunks):
    """
    Cria e salva a vector store global a partir dos documentos enviados.
    """
    global _vector_store_global
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    _vector_store_global = FAISS.from_texts(
        [c.page_content for c in chunks], embeddings
    )
    return _vector_store_global


def get_vector_store():
    """
    Retorna a vector store carregada.
    Lança erro se nenhum documento foi enviado ainda.
    """
    if _vector_store_global is None:
        raise ValueError("Nenhum documento foi carregado ainda.")
    return _vector_store_global
