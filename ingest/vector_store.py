from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def criar_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="ia_labs_docs"
    )
