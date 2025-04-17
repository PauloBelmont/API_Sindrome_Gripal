from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import gdown
import os
import joblib
import numpy as np

# -----------------------------------------------------------------------------
# Configuração inicial e download do modelo (se necessário)
# -----------------------------------------------------------------------------
if not os.path.exists("modelos/melhor_modelo.pkl"):
    url = "https://drive.google.com/uc?id=1JWglKM4BJxkxH5Yc2HLxcOG_gWwCUFsN"
    gdown.download(url, "modelos/melhor_modelo.pkl", quiet=False)
if not os.path.exists("modelos/scaler.pkl"):
    url = "https://drive.google.com/uc?id=1c72HtyYTCMLXw95NNpo9-LjEHRhVstKX"
    gdown.download(url, "modelos/scaler.pkl", quiet=False)

st.set_page_config(page_title="NotificaRR",
                   page_icon="Logo.png",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# Modo Claro e Escuro
# -----------------------------------------------------------------------------
if "modo" not in st.session_state:
    st.session_state["modo"] = "Claro"

def toggle_mode():
    st.session_state["modo"] = "Escuro" if st.session_state["modo"] == "Claro" else "Claro"

cols = st.columns([10, 1])
with cols[1]:
    st.button("🌙" if st.session_state["modo"] == "Claro" else "☀️",
              key="toggle_mode", on_click=toggle_mode)
modo = st.session_state["modo"]

# -----------------------------------------------------------------------------
# Definição de Cores (fundo, texto, botões, bordas, paleta de gráficos)
# -----------------------------------------------------------------------------
if modo == "Escuro":
    bg_color = "#1e1e2f"
    text_color = "#ffffff"
    sidebar_bg = "#252535"
    tab_hover = "#0072BB50"
    border_color = "#50C878"
    # Paleta de gráficos
    confirmados_color = "#00A8A8"
    descartados_color = "#FF6B6B"
    sg_ne_especial_color = "#50C878"
else:
    bg_color = "#ffffff"
    text_color = "#2c2c2c"
    sidebar_bg = "#f7f7f7"
    tab_hover = "#005A8F20"
    border_color = "#0072BB"
    # Paleta de gráficos
    confirmados_color = "#00A8A8"
    descartados_color = "#FF6B6B"
    sg_ne_especial_color = "#50C878"

grid_color = "#4a4a4a" if modo == "Escuro" else "#e0e0e0"
button_bg = "#2d2d44" if modo == "Escuro" else "#f0f0f0"
button_hover = "#3e3e5e" if modo == "Escuro" else "#d6d6d6"

color_map = {
    'Descartado': descartados_color,
    'Confirmado': confirmados_color,
    'Síndrome Gripal Não Especificada': sg_ne_especial_color
}

# -----------------------------------------------------------------------------
# Estilização via CSS
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
body {{ color: {text_color}; }}
.stApp {{ background-color: {bg_color}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}

h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stMetric,  
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
    color: {text_color} !important;
}}

div[data-baseweb="tab-list"] {{
    gap: 1rem;
}}
button[data-baseweb="tab"] {{
    font-size: 1.2rem !important;
    padding: 1rem 2rem !important;
    border-radius: 0.5rem !important;
    transition: all 0.3s ease !important;
}}
button[data-baseweb="tab"]:hover {{
    background-color: {tab_hover} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background-color: {confirmados_color} !important;
    color: white !important;
}}

div.stButton > button {{
    background-color: {button_bg} !important;
    color: {text_color} !important;
    border: none;
    border-radius: 0.5rem;
    transition: background-color 0.3s ease;
}}
div.stButton > button:hover {{
    background-color: {button_hover} !important;
}}

/* Bordas arredondadas para KPIs e Charts */
div[data-testid="stMetric"] {{
    border: 2px solid {border_color} !important;
    border-radius: 1rem !important;
    padding: 1rem !important;
    margin: 0.5rem !important;
}}
div[data-testid="stPlotlyChart"] {{
    border: 2px solid {border_color} !important;
    border-radius: 1rem !important;
    padding: 0.5rem !important;
    margin-bottom: 1rem !important;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Função de Estilização de Gráficos
# -----------------------------------------------------------------------------
def apply_chart_styling(fig):
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=12),
        legend=dict(
            title_font=dict(size=22, color=text_color),
            font=dict(size=16, color=text_color)
        ),
        title=dict(
            font=dict(size=18, color=text_color),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title_font=dict(size=14, color=text_color),
            tickfont=dict(size=12, color=text_color)
        ),
        yaxis=dict(
            title_font=dict(size=14, color=text_color),
            tickfont=dict(size=12, color=text_color)
        )
    )
    return fig

# -----------------------------------------------------------------------------
# Título e Subtítulo
# -----------------------------------------------------------------------------
st.markdown(f"""
<h1 style="text-align: center; font-size: 48px;">Análise de Notificações de Síndromes Gripais</h1>
<h3 style="text-align: center; font-size: 24px;">Dashboard interativo para análise descritiva de casos</h3>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Carregar Dados e Pré-processamento
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("df_preprocessed_new.csv")
    # Converter datas
    for col in ['dataNotificacao', 'dataInicioSintomas', 'dataEncerramento']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    # Criar coluna de status vacinal
    df['status_vacinacao'] = 'Não informado'
    df.loc[df['codigoRecebeuVacina_Sim'] == 1, 'status_vacinacao'] = 'Sim'
    df.loc[df['codigoRecebeuVacina_Não'] == 1, 'status_vacinacao'] = 'Não'
    df.loc[df['codigoRecebeuVacina_Ignorado'] == 1, 'status_vacinacao'] = 'Ignorado'
    return df

df = load_data()

# -----------------------------------------------------------------------------
# Sidebar com Filtros
# -----------------------------------------------------------------------------
st.sidebar.header("Filtros")
st.sidebar.image("Logo.png", caption="NotificaRR")
with st.sidebar:
    resultados = st.multiselect(
        "Resultado do Caso",
        options=df['classificacaoFinal'].unique(),
        default=df['classificacaoFinal'].unique()
    )
    evolucoes = st.multiselect(
        "Evolução do Caso",
        options=df['evolucaoCaso'].unique(),
        default=df['evolucaoCaso'].unique()
    )
    idade_min, idade_max = st.slider(
        "Faixa Etária",
        min_value=int(df['idade'].min()),
        max_value=int(df['idade'].max()),
        value=(int(df['idade'].min()), int(df['idade'].max()))
    )

df_filtrado = df[
    (df['classificacaoFinal'].isin(resultados)) &
    (df['evolucaoCaso'].isin(evolucoes)) &
    (df['idade'] >= idade_min) &
    (df['idade'] <= idade_max)
]

# -----------------------------------------------------------------------------
# Cálculos Comuns para Gráficos
# -----------------------------------------------------------------------------
# Agrupamento etário
df_filtrado['age_group'] = pd.cut(
    df_filtrado['idade'],
    bins=list(range(0, int(df['idade'].max())+10, 10)),
    right=False,
    labels=[f"{i*10} a {(i+1)*10}" for i in range((int(df['idade'].max())//10)+1)]
)

sexo_counts = df_filtrado[['sexo_Feminino', 'sexo_Masculino']].sum().reset_index()
sexo_counts.columns = ['Sexo', 'Contagem']
sexo_counts['Sexo'] = sexo_counts['Sexo'].str.replace('sexo_', '')

prof_counts = df_filtrado[['profissionalSaude_Sim', 'profissionalSaude_Não']].sum().reset_index()
prof_counts.columns = ['Categoria', 'Contagem']
prof_counts['Categoria'] = prof_counts['Categoria'].str.replace('profissionalSaude_', '')

df_evolucao = df_filtrado[df_filtrado['classificacaoFinal'] != 'Descartado']
evolucao_counts = df_evolucao.groupby(['classificacaoFinal', 'evolucaoCaso']).size().reset_index(name='count')
evolucao_counts['percent'] = evolucao_counts.groupby('classificacaoFinal')['count']\
    .transform(lambda x: 100 * x / x.sum())
evolucao_counts['adjusted_percent'] = np.log1p(evolucao_counts['percent'])
evolucao_counts['adjusted_percent'] = evolucao_counts['adjusted_percent'] / evolucao_counts['adjusted_percent'].max() * 100

# -----------------------------------------------------------------------------
# Layout das Abas
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Panorama Geral",
    "👥 Sintomas e Comorbidades",
    "🕒 Análise Temporal",
    "🔮 Predição de Casos"
])

# -----------------------------------------------------------------------------
# Aba 1: Visão Geral
# -----------------------------------------------------------------------------
with tab1:
    st.header("Indicadores Chave e Distribuições")
    cols_kpi = st.columns(4)
    with cols_kpi[0]:
        total_casos = df_filtrado.shape[0]
        st.metric("Total de Casos Analisados", f"{total_casos:,}")
    with cols_kpi[1]:
        obitos = (df_filtrado['evolucaoCaso'] == 'Óbito').sum()
        st.metric("Casos com Evolução para Óbito", f"{obitos} ({obitos/total_casos:.1%})")
    with cols_kpi[2]:
        confirmados = df_filtrado['classificacaoFinal'].str.contains('Confirmado').sum()
        st.metric("Casos Confirmados de COVID-19", f"{confirmados} ({confirmados/total_casos:.1%})")
    with cols_kpi[3]:
        tempo_medio = df_filtrado['dias_entre_sintomas_notificacao'].mean()
        st.metric("Tempo Médio de Notificação", f"{tempo_medio:.1f} dias")

    # Gráficos principais
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(
            df_filtrado,
            names='classificacaoFinal',
            title='Distribuição de Casos por Classificação Final',
            color='classificacaoFinal',
            color_discrete_map=color_map
        )
        st.plotly_chart(apply_chart_styling(fig1), use_container_width=True)
    with col2:
        fig2 = px.bar(
            evolucao_counts,
            x='classificacaoFinal',
            y='adjusted_percent',
            color='evolucaoCaso',
            barmode='group',
            title='Evolução Clínica dos Casos',
            labels={'classificacaoFinal': 'Classificação', 'adjusted_percent': 'Porcentagem', 'evolucaoCaso': 'Evolução'},
            color_discrete_map=color_map
        )
        st.plotly_chart(apply_chart_styling(fig2), use_container_width=True)

    # Demografia
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        fig_sexo = px.pie(sexo_counts, values='Contagem', names='Sexo', title='Distribuição por Sexo')
        st.plotly_chart(apply_chart_styling(fig_sexo), use_container_width=True)
    with demo_col2:
        fig_prof = px.pie(prof_counts, values='Contagem', names='Categoria', title='Profissionais de Saúde Afetados')
        st.plotly_chart(apply_chart_styling(fig_prof), use_container_width=True)

    # Faixa etária
    st.subheader("Distribuição por Faixa Etária")
    fig3 = px.histogram(
        df_filtrado,
        x='age_group',
        color='classificacaoFinal',
        barmode='group',
        title='Distribuição Etária dos Casos',
        labels={'age_group': 'Faixa Etária', 'count': 'Número de Casos'},
        color_discrete_map=color_map
    )
    st.plotly_chart(apply_chart_styling(fig3), use_container_width=True)

# -----------------------------------------------------------------------------
# Aba 2: Sintomas e Condições Pré-existentes
# -----------------------------------------------------------------------------
with tab2:
    st.header("Sintomas e Condições Pré-existentes")
    sintomas = ['Dor de Garganta', 'Coriza', 'Dispneia', 'Distúrbios Olfativos',
                'Distúrbios Gustativos', 'Febre', 'Tosse', 'Dor de Cabeça']
    sintoma_counts = df_filtrado[sintomas].sum().reset_index()
    sintoma_counts.columns = ['Sintoma', 'Contagem']
    fig7 = px.bar(sintoma_counts, x='Contagem', y='Sintoma', orientation='h', title='Sintomas Mais Comuns')
    st.plotly_chart(apply_chart_styling(fig7), use_container_width=True)
    df_filtrado['condicoes'] = df_filtrado['condicoes'].str.strip("[]").str.replace("'", "")
    df_condicoes = df_filtrado['condicoes'].str.get_dummies(sep=', ')
    if not df_condicoes.empty:
        condicao_counts = df_condicoes.sum().sort_values(ascending=False).reset_index()
        condicao_counts.columns = ['Condição', 'Contagem']
        fig8 = px.bar(condicao_counts, x='Contagem', y='Condição', orientation='h', title='Condições Pré-existentes Mais Comuns')
        st.plotly_chart(apply_chart_styling(fig8), use_container_width=True)
    else:
        st.warning("Não há dados suficientes sobre condições pré-existentes.")

# -----------------------------------------------------------------------------
# Aba 3: Análise Temporal
# -----------------------------------------------------------------------------
with tab3:
    st.header("Análise Temporal")
    st.subheader("Casos por Mês de Notificação (2022-2023)")
    if not df_filtrado['dataNotificacao'].isnull().all():
        mask = (df_filtrado['dataNotificacao'] >= '2022-01-01') & (df_filtrado['dataNotificacao'] <= '2023-12-31')
        df_temp = df_filtrado[mask].copy()
        df_temp['mes_notificacao'] = df_temp['dataNotificacao'].dt.to_period('M')
        counts = df_temp.groupby('mes_notificacao').size().reset_index(name='Contagem')
        counts['mes_notificacao'] = counts['mes_notificacao'].dt.to_timestamp()
        all_months = pd.date_range(start='2022-01-01', end='2023-12-31', freq='MS')
        counts = counts.set_index('mes_notificacao').reindex(all_months).fillna(0).reset_index()
        counts.columns = ['Mês', 'Casos']
        fig9 = px.line(counts, x='Mês', y='Casos', title='Evolução Mensal de Notificações', markers=True)
        st.plotly_chart(apply_chart_styling(fig9), use_container_width=True)
    else:
        st.warning("Dados de data de notificação ausentes ou inválidos.")

    st.subheader("Tempo Médio de Notificação por Classificação")
    if 'dias_entre_sintomas_notificacao' in df_filtrado.columns:
        df_tempo = df_filtrado[df_filtrado['dias_entre_sintomas_notificacao'] <= 365].copy()
        tempo_medio = df_tempo.groupby('classificacaoFinal')['dias_entre_sintomas_notificacao']\
            .mean().reset_index(name='Dias')
        fig10 = px.bar(tempo_medio, x='classificacaoFinal', y='Dias', title='Tempo Médio entre Sintomas e Notificação', text_auto='.1f', color='classificacaoFinal', color_discrete_map=color_map)
        st.plotly_chart(apply_chart_styling(fig10), use_container_width=True)
    else:
        st.warning("Dados sobre tempo entre sintomas e notificação não disponíveis.")

# -----------------------------------------------------------------------------
# Aba 4: Previsão de Casos
# -----------------------------------------------------------------------------
with tab4:
    st.header("Previsão de Casos de COVID-19")
    st.markdown("Informe os dados do paciente para realizar a previsão")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            idade = st.number_input("Idade", min_value=0, max_value=120, value=30)
            sexo = st.radio("Sexo", ['Masculino', 'Feminino', 'Indefinido'])
            profiss = st.radio("Profissional de Saúde?", ['Sim', 'Não'])
            vacinado = st.radio("Recebeu vacina?", ['Sim', 'Não', 'Ignorado'])
            total_cond = st.number_input("Número de condições pré-existentes", min_value=0, max_value=10, value=0)
        with col2:
            dias = st.number_input("Dias entre sintomas e notificação", min_value=0, max_value=30, value=3)
            febre = st.checkbox("Febre")
            tosse = st.checkbox("Tosse")
            dor_garg = st.checkbox("Dor de Garganta")
            dispneia = st.checkbox("Dispneia")
            olf = st.checkbox("Distúrbios Olfativos")
            gust = st.checkbox("Distúrbios Gustativos")
            cabeca = st.checkbox("Dor de Cabeça")
            coriza = st.checkbox("Coriza")
            outros = st.checkbox("Outros Sintomas")
            assint = st.checkbox("Assintomático")
        submitted = st.form_submit_button("Realizar Previsão")

    if submitted:
        try:
            model = joblib.load('modelos/melhor_modelo.pkl')
            scaler = joblib.load('modelos/scaler.pkl')
            features = [
                idade,
                1 if profiss=='Não' else 0,
                1 if profiss=='Sim' else 0,
                1 if sexo=='Feminino' else 0,
                1 if sexo=='Indefinido' else 0,
                1 if sexo=='Masculino' else 0,
                1 if vacinado=='Ignorado' else 0,
                1 if vacinado=='Não' else 0,
                1 if vacinado=='Sim' else 0,
                total_cond,
                dias,
                int(dor_garg), int(coriza), int(outros), int(dispneia), int(olf), int(assint), int(gust), int(febre), int(tosse), int(cabeca)
            ]
            scaled = scaler.transform([features])
            pred = model.predict(scaled)[0]
            proba = np.max(model.predict_proba(scaled)[0])*100
            st.subheader("Resultado da Previsão:")
            if pred=='Confirmado': st.error(f"**Resultado:** {pred} (Confiança: {proba:.1f}%)")
            else: st.success(f"**Resultado:** {pred} (Confiança: {proba:.1f}%)")
            imps = model.feature_importances_
            names = ['Idade','Profissional Não','Profissional Sim','Sexo Fem','Sexo Ind','Sexo Masc','Vac Ign','Vac Não','Vac Sim','Total Cond','Dias','Dor Garg','Coriza','Outros','Dispneia','Olf','Assint','Gust','Febre','Tosse','Cabeca']
            st.markdown("### Fatores mais relevantes para a decisão:")
            for i in np.argsort(imps)[::-1][:5]: st.write(f"- {names[i]} ({imps[i]*100:.1f}%)")
        except Exception as e:
            st.error(f"Erro na previsão: {str(e)}")

# -----------------------------------------------------------------------------
# Rodapé
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("Dashboard desenvolvido para análise de dados de COVID-19 e síndromes gripais")