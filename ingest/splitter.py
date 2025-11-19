from langchain_text_splitters import RecursiveCharacterTextSplitter

def dividir_documentos(textos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    chunks = []
    for t in textos:
        chunks.extend(splitter.split_text(t))

    return chunks
