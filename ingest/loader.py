from pypdf import PdfReader
from docx import Document

def carregar_documentos(files):
    textos = []

    for f in files:
        if f.name.endswith(".pdf"):
            reader = PdfReader(f)
            texto = ""
            for page in reader.pages:
                texto += page.extract_text() or ""
            textos.append(texto)

        elif f.name.endswith(".docx"):
            doc = Document(f)
            texto = "\n".join([p.text for p in doc.paragraphs])
            textos.append(texto)

        elif f.name.endswith(".txt"):
            textos.append(f.read().decode("utf-8"))

    return textos
