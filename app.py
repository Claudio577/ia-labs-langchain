import streamlit as st

# ================================
# IMPORTAR MÓDULOS INTERNOS
# ================================
from ingest.loader import carregar_documentos
from ingest.splitter import dividir_documentos
from ingest.vector_store import criar_vector_store

from agents.agent_corporativo import criar_agente_corporativo
from agents.agente_executivo import criar_agente_executivo
from agents.agente_juridico import criar_agente_juridico
from agents.agente_financeiro import criar_agente_financeiro
from agents.agente_compliance import criar_agente_compliance

# ---------------------------------
# CONFIGURAÇÃO DO STREAMLIT
# ---------------------------------
st.set_page_config(
    page_title="Inteligência Corporativa",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown("""
<h1 style="color:#1E88E5; text-align:center; font-weight:600;">
INTELIGÊNCIA CORPORATIVA • AGENTES PROFISSIONAIS
</h1>
<p style="text-align:center; font-size:17px; color:#444;">
Sistema modular de análise assistida por IA, RAG corporativo e agentes especializados.
</p>
<hr>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("🔎 Seleção de Agente")

agente_escolhido = st.sidebar.selectbox(
    "Escolha o agente:",
    [
        "Assistente Corporativo",
        "Agente Executivo",
        "Agente Jurídico",
        "Agente Financeiro",
        "Agente de Compliance"
    ]
)

def carregar_agente():
    if agente_escolhido == "Assistente Corporativo":
        return criar_agente_corporativo()
    elif agente_escolhido == "Agente Executivo":
        return criar_agente_executivo()
    elif agente_escolhido == "Agente Jurídico":
        return criar_agente_juridico()
    elif agente_escolhido == "Agente Financeiro":
        return criar_agente_financeiro()
    elif agente_escolhido == "Agente de Compliance":
        return criar_agente_compliance()

# ---------------------------------
# UPLOAD DE DOCUMENTOS
# ---------------------------------
st.subheader("📄 Enviar Documentos")

uploaded_files = st.file_uploader(
    "Selecione arquivos PDF, DOCX ou TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processando documentos..."):
        docs = carregar_documentos(uploaded_files)
        chunks = dividir_documentos(docs)
        criar_vector_store(chunks)
    st.success("Base de conhecimento atualizada com sucesso!")
    st.markdown("---")

# ---------------------------------
# ÁREA DE PERGUNTAS
# ---------------------------------
st.subheader(f"💬 Conversar com: **{agente_escolhido}**")

query = st.text_area(
    "Digite sua pergunta ou descrição da tarefa:",
    height=120
)

if st.button("Enviar"):
    if not query.strip():
        st.warning("Digite uma pergunta antes de enviar.")
    else:
        agente = carregar_agente()
        
        with st.spinner("Processando com IA..."):
            try:
                resposta = agente.run(query)
            except Exception as e:
                resposta = f"Erro ao processar: {e}"

        st.markdown("### 🔍 Resposta")
        st.write(resposta)

