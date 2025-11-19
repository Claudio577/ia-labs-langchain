import streamlit as st

from ingest.loader import carregar_documentos
from ingest.splitter import dividir_documentos
from ingest.vector_store import criar_vector_store

from agents.agent_corporativo import criar_agente_corporativo
from agents.agente_executivo import criar_agente_executivo
from agents.agente_juridico import criar_agente_juridico
from agents.agente_financeiro import criar_agente_financeiro
from agents.agente_compliance import criar_agente_compliance

st.set_page_config(page_title="IA Labs - Agentes Corporativos", layout="wide")

st.title("⚙️ IA-Labs — Inteligência Corporativa")

# -------------------
# Seletor de agente
# -------------------
agente_nome = st.sidebar.selectbox(
    "Escolher agente",
    [
        "Assistente Corporativo",
        "Agente Executivo",
        "Agente Jurídico",
        "Agente Financeiro",
        "Agente de Compliance"
    ]
)

def carregar_agente():
    if agente_nome == "Assistente Corporativo":
        return criar_agente_corporativo()
    elif agente_nome == "Agente Executivo":
        return criar_agente_executivo()
    elif agente_nome == "Agente Jurídico":
        return criar_agente_juridico()
    elif agente_nome == "Agente Financeiro":
        return criar_agente_financeiro()
    elif agente_nome == "Agente de Compliance":
        return criar_agente_compliance()

# -------------------------
# Upload de documentos
# -------------------------
st.subheader("📄 Enviar documentos")
files = st.file_uploader("Selecione arquivos", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if files:
    docs = carregar_documentos(files)
    chunks = dividir_documentos(docs)
    criar_vector_store(chunks)
    st.success("Base atualizada com sucesso!")

# -------------------------
# Chat com o agente
# -------------------------
st.subheader(f"💬 Conversar com: {agente_nome}")
query = st.text_area("Digite sua pergunta")

if st.button("Enviar"):
    agente = carregar_agente()
    resposta = agente.run(query)
    st.write("### Resposta")
    st.write(resposta)
