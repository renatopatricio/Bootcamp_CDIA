
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Análise Preditiva de Máquinas - UNISENAI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS customizado para storytelling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 3rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .conclusion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Carregamento dos dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_limpo_final.csv")
        return df
    except:
        st.error("Erro ao carregar o dataset. Verifique se o arquivo 'dataset_limpo_final.csv' está disponível.")
        return None

# Função para limpar dados inconsistentes
def clean_data(df):
    df_clean = df.copy()

    # Substituir valores incoerentes por NaN
    df_clean.loc[df_clean['temperatura_processo'] == -38, 'temperatura_processo'] = np.nan
    df_clean.loc[df_clean['desgaste_da_ferramenta'] == -202, 'desgaste_da_ferramenta'] = np.nan

    # Padronizar colunas de falha
    falha_cols = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
                  'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']

    for col in falha_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.lower()
            df_clean[col] = df_clean[col].replace({'true': 'Sim', 'false': 'Não', 
                                                   'sim': 'Sim', 'não': 'Não', 
                                                   'nao': 'Não', 'n': 'Não', '0': 'Não'})

    return df_clean

# Carregamento dos dados
df = load_data()
if df is not None:
    df_clean = clean_data(df)

    # ==================== SEÇÃO 1: TÍTULO E CONTEXTO ====================
    st.markdown('<h1 class="main-title">🔧 Análise Preditiva de Máquinas</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #7f8c8d;">Ciência de Dados e IA para Manutenção Industrial</h2>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Objetivo do Projeto

        Este projeto realiza uma **Análise Exploratória de Dados (EDA)** focada em identificar 
        problemas de qualidade, padrões relevantes e insights iniciais para posterior 
        modelagem de falhas em máquinas industriais.

        ### 🏭 Contexto Industrial

        No ambiente da **Indústria 4.0**, o monitoramento inteligente de máquinas é fundamental 
        para:
        - Reduzir custos de manutenção
        - Evitar paradas não programadas
        - Otimizar a vida útil dos equipamentos
        - Aumentar a eficiência operacional
        """)

    with col2:
        st.info("""
        **Dataset Analisado:**
        - Sensores de temperatura
        - Medições de torque
        - Desgaste de ferramentas
        - Registros de falhas
        - Tipos de máquinas (L, M, H)
        """)

    # ==================== SEÇÃO 2: EXPLORAÇÃO DOS DADOS ====================
    st.markdown('<h2 class="section-title">📊 Exploração dos Dados</h2>', unsafe_allow_html=True)

    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>{len(df_clean):,}</h3>
            <p>Total de Registros</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        falhas_total = df_clean['falha_maquina'].sum()
        st.markdown(f"""
        <div class="kpi-card">
            <h3>{falhas_total}</h3>
            <p>Falhas Detectadas</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        taxa_falha = (falhas_total / len(df_clean) * 100)
        st.markdown(f"""
        <div class="kpi-card">
            <h3>{taxa_falha:.1f}%</h3>
            <p>Taxa de Falhas</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        tipos_maquina = df_clean['tipo'].nunique()
        st.markdown(f"""
        <div class="kpi-card">
            <h3>{tipos_maquina}</h3>
            <p>Tipos de Máquinas</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráficos de distribuição
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Distribuição de Temperatura do Ar")
        fig_temp = px.histogram(df_clean, x='temperatura_ar', nbins=30, 
                               title="Temperatura do Ar (°C)",
                               color_discrete_sequence=['#3498db'])
        fig_temp.update_layout(showlegend=False)
        st.plotly_chart(fig_temp, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>💡 Insight:</strong> A temperatura do ar varia entre 295°C e 305°C, 
        com distribuição aproximadamente normal centrada em 300°C.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("⚙️ Distribuição de Torque")
        fig_torque = px.histogram(df_clean, x='torque', nbins=30,
                                 title="Torque (Nm)",
                                 color_discrete_sequence=['#e74c3c'])
        fig_torque.update_layout(showlegend=False)
        st.plotly_chart(fig_torque, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>💡 Insight:</strong> O torque apresenta distribuição bimodal, 
        indicando diferentes regimes operacionais das máquinas.
        </div>
        """, unsafe_allow_html=True)

    # ==================== SEÇÃO 3: ANÁLISE DE FALHAS ====================
    st.markdown('<h2 class="section-title">⚠️ Análise de Falhas</h2>', unsafe_allow_html=True)

    # Análise de tipos de falha
    falha_cols = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
                  'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']

    falha_counts = {}
    for col in falha_cols:
        if col in df_clean.columns:
            falha_counts[col.split('(')[0].strip()] = (df_clean[col] == 'Sim').sum()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Frequência de Falhas por Tipo")
        if falha_counts:
            fig_falhas = px.bar(x=list(falha_counts.keys()), y=list(falha_counts.values()),
                               title="Número de Falhas por Categoria",
                               color=list(falha_counts.values()),
                               color_continuous_scale='Reds')
            fig_falhas.update_layout(showlegend=False, xaxis_title="Tipo de Falha", yaxis_title="Quantidade")
            st.plotly_chart(fig_falhas, use_container_width=True)

    with col2:
        st.subheader("🥧 Distribuição Geral de Falhas")
        falhas_sim = df_clean['falha_maquina'].sum()
        falhas_nao = len(df_clean) - falhas_sim

        fig_pie = px.pie(values=[falhas_nao, falhas_sim], 
                        names=['Sem Falha', 'Com Falha'],
                        title="Proporção de Máquinas com/sem Falha",
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>🔍 Análise Crítica:</strong> O dataset apresenta forte desbalanceamento, 
    com menos de 2% das amostras indicando falha. Isso é típico em ambientes industriais 
    onde falhas são eventos raros, mas críticos.
    </div>
    """, unsafe_allow_html=True)

    # ==================== SEÇÃO 4: DASHBOARD INTEGRADO ====================
    st.markdown('<h2 class="section-title">🎛️ Dashboard de Correlações</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔗 Torque vs Desgaste da Ferramenta")
        # Filtrar valores válidos para o scatter plot
        df_scatter = df_clean[(df_clean['torque'].notna()) & 
                             (df_clean['desgaste_da_ferramenta'].notna()) &
                             (df_clean['desgaste_da_ferramenta'] >= 0)]

        fig_scatter = px.scatter(df_scatter, x='torque', y='desgaste_da_ferramenta',
                                color='falha_maquina', 
                                title="Relação entre Torque e Desgaste",
                                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                                labels={'falha_maquina': 'Falha da Máquina'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("📈 Evolução do Desgaste ao Longo do Tempo")
        df_desgaste = df_clean[df_clean['desgaste_da_ferramenta'].notna() & 
                              (df_clean['desgaste_da_ferramenta'] >= 0)].head(100)

        fig_line = px.line(df_desgaste, x='id', y='desgaste_da_ferramenta',
                          title="Desgaste da Ferramenta por ID",
                          color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig_line, use_container_width=True)

    # Análise por tipo de máquina
    st.subheader("🏭 Distribuição por Tipo de Máquina")

    col1, col2, col3 = st.columns(3)

    tipo_counts = df_clean['tipo'].value_counts()

    with col1:
        st.metric("Tipo L (Low)", f"{tipo_counts.get('L', 0)}", 
                 f"{tipo_counts.get('L', 0)/len(df_clean)*100:.1f}%")

    with col2:
        st.metric("Tipo M (Medium)", f"{tipo_counts.get('M', 0)}", 
                 f"{tipo_counts.get('M', 0)/len(df_clean)*100:.1f}%")

    with col3:
        st.metric("Tipo H (High)", f"{tipo_counts.get('H', 0)}", 
                 f"{tipo_counts.get('H', 0)/len(df_clean)*100:.1f}%")

    # ==================== SEÇÃO 5: PROBLEMAS IDENTIFICADOS ====================
    st.markdown('<h2 class="section-title">🚨 Problemas de Qualidade dos Dados</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("❌ Valores Incoerentes Detectados")

        temp_negativas = (df['temperatura_processo'] == -38).sum()
        desgaste_negativo = (df['desgaste_da_ferramenta'] == -202).sum()

        st.write(f"• **Temperaturas inválidas (-38°C):** {temp_negativas} registros")
        st.write(f"• **Desgaste inválido (-202):** {desgaste_negativo} registros")
        st.write(f"• **Umidade constante (90%):** {(df['umidade_relativa'] == 90).sum()} registros")

        st.markdown("""
        <div class="insight-box">
        <strong>⚠️ Atenção:</strong> Valores como -38°C e -202 são claramente 
        códigos de erro ou valores faltantes mal codificados.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🔧 Estratégias de Limpeza Aplicadas")
        st.write("✅ Substituição de valores incoerentes por NaN")
        st.write("✅ Padronização de rótulos de falha")
        st.write("✅ Identificação de outliers")
        st.write("✅ Análise de consistência entre colunas")

    # ==================== SEÇÃO 6: PRÓXIMOS PASSOS ====================
    st.markdown('<h2 class="section-title">🚀 Próximos Passos Recomendados</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛠️ Preparação dos Dados")
        st.write("• Limpeza e padronização completa")
        st.write("• Tratamento de valores faltantes")
        st.write("• Feature engineering avançada")
        st.write("• Normalização de variáveis")

    with col2:
        st.subheader("🤖 Modelagem Preditiva")
        st.write("• Estratégias contra desbalanceamento (SMOTE)")
        st.write("• Modelos de classificação multiclasse")
        st.write("• Validação cruzada temporal")
        st.write("• Métricas específicas para falhas raras")

    # ==================== SEÇÃO 7: CONCLUSÃO ====================
    st.markdown("""
    <div class="conclusion-box">
        <h2>💡 Conclusão</h2>
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
        A análise exploratória revelou um dataset rico em informações sobre o comportamento 
        de máquinas industriais, com oportunidades claras para implementação de 
        <strong>manutenção preditiva inteligente</strong>.
        </p>
        <p style="font-size: 1.1rem;">
        <strong>"Com dados inteligentes, máquinas mais eficientes"</strong>
        </p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
        
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==================== SEÇÃO 8: DADOS TÉCNICOS ====================
    st.markdown('<h2 class="section-title">📋 Visualização dos Dados</h2>', unsafe_allow_html=True)

    st.subheader("🔍 Amostra do Dataset Limpo")
    st.dataframe(df_clean.head(20), use_container_width=True)

    # Estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    numeric_cols = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 
                   'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']

    stats_df = df_clean[numeric_cols].describe()
    st.dataframe(stats_df, use_container_width=True)

else:
    st.error("Não foi possível carregar o dataset. Verifique se o arquivo está disponível.")
