import streamlit as st
from streamlit_option_menu import option_menu

# INGESTÃO
from ingest.loader import carregar_documentos
from ingest.splitter import dividir_documentos
from ingest.vector_store import criar_vector_store

# AGENTES
from agents.agent_corporativo import criar_agente_corporativo
from agents.agente_executivo import criar_agente_executivo
from agents.agente_juridico import criar_agente_juridico
from agents.agente_financeiro import criar_agente_financeiro
from agents.agente_compliance import criar_agente_compliance

# CSS CORPORATIVO NEUTRO
st.markdown("""
<style>

body {
    background-color: #f5f6f7;
    font-family: 'Inter', sans-serif;
}

/* Título principal */
h1 {
    color: #2E3A59;
    font-weight: 700;
}

/* Subtítulos */
h2, h3 {
    color: #405070;
    font-weight: 600;
}

/* Caixas */
section {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e1e4e8;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.04);
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# CONFIG APP
# ======================================================
st.set_page_config(
    page_title="Inteligência Corporativa",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Inteligência Corporativa — Plataforma de Análise Documental")


# ======================================================
# MENU LATERAL — Abas corporativas
# ======================================================
with st.sidebar:
    selected = option_menu(
        "Menu Principal",
        [
            "📄 Base de Conhecimento",
            "🤖 Assistentes Inteligentes",
            "📊 Insights Automáticos",
            "📝 Relatório PDF"
        ],
        icons=["file-earmark-text", "robot", "graph-up", "filetype-pdf"],
        menu_icon="grid",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "nav-link": {"color": "#2E3A59"},
            "nav-link-selected": {"background-color": "#E8EEF6", "color": "#1C3D6C"},
        }
    )


# ======================================================
# TELA 1 — BASE DE CONHECIMENTO
# ======================================================
if selected == "📄 Base de Conhecimento":
    st.header("📄 Upload de Documentos")

    files = st.file_uploader(
        "Envie PDFs, DOCXs ou TXTs",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if files:
        with st.spinner("Processando documentos..."):
            docs = carregar_documentos(files)
            chunks = dividir_documentos(docs)
            criar_vector_store(chunks)

        st.success("✔ Base de conhecimento atualizada com sucesso!")


# ======================================================
# TELA 2 — ASSISTENTES
# ======================================================
elif selected == "🤖 Assistentes Inteligentes":

    st.header("🤖 Assistentes Inteligentes")

    agente_nome = st.selectbox(
        "Selecione o agente desejado:",
        [
            "Assistente Corporativo Geral",
            "Agente Executivo",
            "Agente Jurídico (ContractAI)",
            "Agente Financeiro",
            "Agente de Compliance"
        ]
    )

    if agente_nome == "Assistente Corporativo Geral":
        agente = criar_agente_corporativo()
    elif agente_nome == "Agente Executivo":
        agente = criar_agente_executivo()
    elif agente_nome == "Agente Jurídico (ContractAI)":
        agente = criar_agente_juridico()
    elif agente_nome == "Agente Financeiro":
        agente = criar_agente_financeiro()
    elif agente_nome == "Agente de Compliance":
        agente = criar_agente_compliance()

    st.subheader(f"💬 Conversar com: {agente_nome}")

    query = st.text_area("Digite sua pergunta:")

    if st.button("Enviar Pergunta"):
        with st.spinner("Gerando análise..."):
            resposta = agente.run(query)

        st.subheader("📌 Resposta do Sistema")
        st.write(resposta)



# ======================================================
# TELA 3 — INSIGHTS (analytics)
# ======================================================
elif selected == "📊 Insights Automáticos":
    st.header("📊 Insights Avançados")
    st.info("Em breve: extração automática de tópicos importantes, riscos e recomendações.")


# ======================================================
# TELA 4 — RELATÓRIO PDF
# ======================================================
elif selected == "📝 Relatório PDF":
    st.header("📝 Gerar Relatório PDF")
    st.info("Em breve: gerar relatório corporativo consolidado dos documentos + análises.")

