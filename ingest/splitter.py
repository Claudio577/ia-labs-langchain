from langchain.text_splitter import RecursiveCharacterTextSplitter

def dividir_documentos(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)
