from openai import OpenAI
from langchain_community.vectorstores import Chroma

client = OpenAI()

_vector_store_global = None

def criar_vector_store(chunks):
    """
    Cria a base vetorial usando o cliente oficial da OpenAI,
    gerando embeddings manualmente.
    """
    global _vector_store_global

    textos = [
        c.page_content if hasattr(c, "page_content") else str(c)
        for c in chunks
    ]

    # Gera embeddings manualmente
    embeddings = []
    for t in textos:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=t
        )
        embeddings.append(resp.data[0].embedding)

    # Cria a base vetorial Chroma
    _vector_store_global = Chroma.from_embeddings(
        embeddings=embeddings,
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(textos))],
        ids=[str(i) for i in range(len(textos))],
        collection_name="ia_labs_docs"
    )

    return _vector_store_global


def get_vector_store():
    if _vector_store_global is None:
        raise ValueError("Nenhum documento foi carregado ainda.")
    return _vector_store_global

