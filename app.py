import streamlit as st

# INGESTÃO
from ingest.loader import carregar_documentos
from ingest.splitter import dividir_documentos
from ingest.vector_store import criar_vector_store

# AGENTES IA-LABS
from agents.agent_corporativo import criar_agente_corporativo
from agents.agente_executivo import criar_agente_executivo
from agents.agente_juridico import criar_agente_juridico
from agents.agente_financeiro import criar_agente_financeiro
from agents.agente_compliance import criar_agente_compliance

# CONFIGURAÇÃO DO APP
st.set_page_config(
    page_title="Inteligência Corporativa",
    layout="wide",
    page_icon="⚙️"
)

# =======================
# CABEÇALHO E LAYOUT
# =======================
st.markdown("""
<h1 style='text-align:center; color:#1E88E5; margin-bottom:0'>
Inteligência Corporativa com LangChain
</h1>
<p style='text-align:center; font-size:18px; margin-top:0'>
Sistema profissional de análise corporativa, RAG e agentes inteligentes.
</p>
<hr>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR — ESCOLHA DO AGENTE IA-LABS
# ==========================================================
st.sidebar.title("🤖 Escolher Agente IA-Labs")

agente_escolhido = st.sidebar.selectbox(
    "Selecione o tipo de análise:",
    [
        "Assistente Corporativo Geral",
        "Agente Executivo",
        "Agente Jurídico (ContractAI)",
        "Agente Financeiro",
        "Agente de Compliance"
    ]
)

def carregar_agente():
    if agente_escolhido == "Assistente Corporativo Geral":
        return criar_agente_corporativo()
    elif agente_escolhido == "Agente Executivo":
        return criar_agente_executivo()
    elif agente_escolhido == "Agente Jurídico (ContractAI)":
        return criar_agente_juridico()
    elif agente_escolhido == "Agente Financeiro":
        return criar_agente_financeiro()
    elif agente_escolhido == "Agente de Compliance":
        return criar_agente_compliance()

# ==========================================================
# ENVIO DE DOCUMENTOS
# ==========================================================
st.subheader("📄 Enviar Documentos para Análise")

files = st.file_uploader(
    "Selecione PDFs, DOCXs ou TXTs:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if files:
    with st.spinner("🔄 Processando documentos..."):
        docs = carregar_documentos(files)
        chunks = dividir_documentos(docs)
        criar_vector_store(chunks)
    st.success("✔ Base de conhecimento atualizada com sucesso!")

st.markdown("---")

# ==========================================================
# ÁREA DE PERGUNTAS (CHAT)
# ==========================================================
st.subheader(f"💬 Conversar com: **{agente_escolhido}**")

query = st.text_area("Digite sua pergunta ou solicitação:")

if st.button("Enviar Pergunta"):
    if not query:
        st.warning("Digite uma pergunta antes de enviar.")
    else:
        agente = carregar_agente()

        with st.spinner("🤖 IA-Labs analisando..."):
            resposta = agente.run(query)

        st.markdown("### 📌 Resposta IA-Labs")
        st.write(resposta)

