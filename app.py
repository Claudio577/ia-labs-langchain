import streamlit as st

# ===============================
# IMPORTS — INGESTÃO
# ===============================
from ingest.loader import carregar_documentos
from ingest.splitter import dividir_documentos
from ingest.vector_store import criar_vector_store

# ===============================
# IMPORTS — AGENTES IA-LABS
# ===============================
from agents.agent_corporativo import criar_agente_corporativo
from agents.agente_executivo import criar_agente_executivo
from agents.agente_juridico import criar_agente_juridico
from agents.agente_financeiro import criar_agente_financeiro
from agents.agente_compliance import criar_agente_compliance


# ===============================
# CONFIGURAÇÃO DO STREAMLIT
# ===============================
st.set_page_config(
    page_title="IA-Labs — Inteligência Corporativa",
    layout="wide",
    page_icon="⚙️"
)

# ===============================
# ESTILO CORPORATIVO NEUTRO
# ===============================
st.markdown("""
<style>

body {
    background-color: #f5f7fa;
    font-family: 'Arial', sans-serif;
}

h1, h2, h3 {
    color: #1e293b;
    font-weight: 600;
}

.section-box {
    padding: 18px;
    background: white;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# CABEÇALHO
# ===============================
st.markdown("""
<h1 style='text-align:center; margin-bottom:0'>
⚙️ IA-Labs — Inteligência Corporativa
</h1>

<p style='text-align:center; font-size:17px; color:#475569; margin-top:4px'>
Sistema profissional de análise documental e geração de insights via LangChain + OpenAI.
</p>
<hr>
""", unsafe_allow_html=True)


# ===============================
# SIDEBAR — ESCOLHA DO AGENTE
# ===============================
st.sidebar.title("🤖 Selecione o Agente IA-Labs")

agente_escolhido = st.sidebar.selectbox(
    "Escolha o tipo de análise:",
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


# ===============================
# SEÇÃO DE ENVIO DE DOCUMENTOS
# ===============================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("📄 Enviar documentos para análise")

files = st.file_uploader(
    "Envie arquivos PDF, DOCX ou TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if files:
    with st.spinner("🔄 Processando documentos..."):
        docs = carregar_documentos(files)
        chunks = dividir_documentos(docs)
        criar_vector_store(chunks)
    st.success("✔ Base de conhecimento atualizada!")
st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# ÁREA DE PERGUNTAS (CHAT)
# ===============================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader(f"💬 Interação com: **{agente_escolhido}**")

query = st.text_area("Digite uma pergunta ou solicitação:")

if st.button("Enviar"):
    if not query:
        st.warning("Digite uma pergunta primeiro.")
    else:
        agente = carregar_agente()
        with st.spinner("🤖 IA-Labs analisando..."):
            resposta = agente.run(query)

        st.markdown("### 🧠 Resposta do Agente")
        st.write(resposta)

st.markdown("</div>", unsafe_allow_html=True)
