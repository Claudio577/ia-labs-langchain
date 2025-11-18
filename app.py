import streamlit as st
from ingest.loader import carregar_documentos
from ingest.splitter import dividir_documentos
from ingest.vector_store import criar_vector_store
from agents.agent_corporativo import criar_agente_corporativo

st.set_page_config(
    page_title="IA-Labs — Assistente Corporativo",
    layout="wide",
    page_icon="⚙️"
)

st.title("⚙️ IA-Labs — Assistente Corporativo com LangChain")

st.subheader("📄 Envie Documentos")
files = st.file_uploader(
    "Envie PDF/DOCX/TXT para análise inteligente",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if files:
    st.info("📚 Processando documentos...")
    docs = carregar_documentos(files)
    chunks = dividir_documentos(docs)
    criar_vector_store(chunks)
    st.success("✔ Base de conhecimento atualizada!")

st.subheader("💬 Converse com o Assistente IA-Labs")
query = st.text_area("Digite sua pergunta:")

if st.button("Enviar"):
    agent = criar_agente_corporativo()
    resposta = agent.run(query)
    st.write(resposta)
