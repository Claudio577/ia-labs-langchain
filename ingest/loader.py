from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
from io import BytesIO

def carregar_documentos(uploaded_files):
    documentos = []

    for uploaded in uploaded_files:
        file_bytes = uploaded.read()

        if uploaded.type == "application/pdf":
            loader = PyPDFLoader(BytesIO(file_bytes))
        elif "word" in uploaded.type:
            loader = Docx2txtLoader(BytesIO(file_bytes))
        else:
            loader = TextLoader(BytesIO(file_bytes))

        documentos.extend(loader.load())

    return documentos
