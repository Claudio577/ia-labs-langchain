from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config import CHROMA_PATH, EMBEDDING_MODEL

def criar_vector_store(docs):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectordb

def carregar_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
