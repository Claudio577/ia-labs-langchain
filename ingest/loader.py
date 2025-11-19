from pypdf import PdfReader
from docx import Document

def carregar_documentos(files):
    documentos = []

    for f in files:
        if f.name.endswith(".pdf"):
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documentos.append(text)

        elif f.name.endswith(".docx"):
            doc = Document(f)
            text = "\n".join([p.text for p in doc.paragraphs])
            documentos.append(text)

        elif f.name.endswith(".txt"):
            documentos.append(f.read().decode("utf-8"))

    return documentos
