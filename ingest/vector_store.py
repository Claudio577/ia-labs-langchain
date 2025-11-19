from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# -----------------------------------------------
# Armazena a Vector Store globalmente
# -----------------------------------------------
_vector_store_global = None


# -----------------------------------------------
# Criação da base vetorial
# -----------------------------------------------
def criar_vector_store(chunks):
    """
    Cria e salva a vector store global a partir dos documentos enviados.
    """
    global _vector_store_global

    embeddings = OpenAIEmbeddings()

    # Extrai apenas o texto dos chunks (List[str])
    textos = [c.page_content for c in chunks]

    _vector_store_global = FAISS.from_texts(textos, embeddings)

    return _vector_store_global


# -----------------------------------------------
# Retorno da vector store
# -----------------------------------------------
def get_vector_store():
    """
    Retorna a vector store carregada.
    Levanta erro caso nenhum documento tenha sido enviado ainda.
    """
    if _vector_store_global is None:
        raise ValueError("Nenhum documento foi carregado ainda.")
    return _vector_store_global

