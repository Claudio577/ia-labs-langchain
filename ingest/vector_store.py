from langchain_community.vectorstores import Chroma


_vector_store_global = None


def criar_vector_store(chunks):
    """
    Cria a base vetorial usando embeddings da OpenAI gerados na hora.
    """
    global _vector_store_global

    # IMPORTA AQUI (não no topo!)
    from openai import OpenAI
    client = OpenAI()  # agora a chave já foi carregada pelo Streamlit

    # Extrai textos
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

    # Cria Chroma
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

