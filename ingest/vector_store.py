from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

_vector_store_global = None

def criar_vector_store(chunks):
    """
    Cria a base vetorial usando ChromaDB.
    """
    global _vector_store_global

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    textos = [
        c.page_content if hasattr(c, "page_content") else str(c)
        for c in chunks
    ]

    _vector_store_global = Chroma.from_texts(
        texts=textos,
        embedding=embeddings,
        collection_name="ia_labs_docs"
    )

    return _vector_store_global


def get_vector_store():
    """
    Retorna a vector store.
    """
    if _vector_store_global is None:
        raise ValueError("Nenhum documento foi carregado ainda.")
    return _vector_store_global

