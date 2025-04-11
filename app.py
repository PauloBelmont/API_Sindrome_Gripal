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
if not os.path.exists("melhor_modelo.pkl"):
    url = "https://drive.google.com/uc?id=1JWglKM4BJxkxH5Yc2HLxcOG_gWgCUFsN"
    gdown.download(url, "melhor_modelo.pkl", quiet=False)

st.set_page_config(page_title="NotificaRR",
                   page_icon="🦠",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# Controle de Modo com Toggle via Botão com Ícones no Canto Superior Direito
# -----------------------------------------------------------------------------
if "modo" not in st.session_state:
    st.session_state["modo"] = "Noturno"

def toggle_mode():
    st.session_state["modo"] = "Diurno" if st.session_state["modo"] == "Noturno" else "Noturno"

cols = st.columns([10, 1])
with cols[1]:
    st.button("🌙" if st.session_state["modo"] == "Noturno" else "☀️",
              key="toggle_mode", on_click=toggle_mode)
modo = st.session_state["modo"]

# -----------------------------------------------------------------------------
# Definição de Cores e Estilos conforme o Modo (Atualizado)
# -----------------------------------------------------------------------------
if modo == "Noturno":
    bg_color = "#1e1e2f"
    text_color = "#ffffff"  # Branco para melhor contraste
    confirmados_color = "#0072BB"
    sg_ne_especial_color = "#50C878"
    descartados_color = "#ff7c7c"
    grid_color = "#4a4a4a"
else:
    bg_color = "#ffffff"
    text_color = "#2c2c2c"   # Cinza escuro para melhor legibilidade
    confirmados_color = "#005A8F"
    sg_ne_especial_color = "#3AA17E"
    descartados_color = "#cc5c5c"
    grid_color = "#e0e0e0"

color_map = {
    'Descartado': descartados_color,
    'Confirmado': confirmados_color,
    'Síndrome Gripal Não Especificada': sg_ne_especial_color
}

# -----------------------------------------------------------------------------
# CSS Global Atualizado
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
body {{ color: {text_color}; }}
.stApp {{ background-color: {bg_color}; }}

/* Aplicar cor do texto para todos elementos */
h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stMetric, 
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
    color: {text_color} !important;
}}

/* Abas maiores e mais visíveis */
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
    background-color: {confirmados_color}20 !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background-color: {confirmados_color} !important;
    color: white !important;
}}

/* Melhoria nos filtros da sidebar */
[data-testid="stSidebar"] label {{
    font-size: 1.1rem !important;
    color: {text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Função de Estilização de Gráficos (Atualizada)
# -----------------------------------------------------------------------------
def apply_chart_styling(fig):
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=12),  # Reduzido de 14
        legend=dict(
            title_font=dict(size=22, color=text_color),  # Reduzido de 26
            font=dict(size=16, color=text_color)         # Reduzido de 20
        ),
        title=dict(
            font=dict(size=18, color=text_color),        # Reduzido de 20
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title_font=dict(size=14, color=text_color), # Reduzido de 16
            tickfont=dict(size=12, color=text_color)     # Reduzido de 14
        ),
        yaxis=dict(
            title_font=dict(size=14, color=text_color), # Reduzido de 16
            tickfont=dict(size=12, color=text_color)     # Reduzido de 14
        )
    )
    return fig

# -----------------------------------------------------------------------------
# Título e Subtítulo Centralizados do Dashboard
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
    date_cols = ['dataNotificacao', 'dataInicioSintomas', 'dataEncerramento']
    for col in date_cols:
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
bins = list(range(0, int(df['idade'].max()) + 10, 10))
labels = [f"{bins[i]} a {bins[i+1]}" for i in range(len(bins)-1)]
df_filtrado['age_group'] = pd.cut(df_filtrado['idade'], bins=bins, right=False, labels=labels)

# Cálculo de status vacinal (já está no pré-processamento)

# Cálculo de distribuição por sexo
sexo_counts = df_filtrado[['sexo_Feminino', 'sexo_Masculino', 'sexo_Indefinido']].sum().reset_index()
sexo_counts.columns = ['Sexo', 'Contagem']
sexo_counts['Sexo'] = sexo_counts['Sexo'].str.replace('sexo_', '')
sexo_counts = sexo_counts[sexo_counts['Sexo'] != 'Indefinido']

# Cálculo de profissionais de saúde
prof_counts = df_filtrado[['profissionalSaude_Sim', 'profissionalSaude_Não']].sum().reset_index()
prof_counts.columns = ['Categoria', 'Contagem']
prof_counts['Categoria'] = prof_counts['Categoria'].str.replace('profissionalSaude_', '')

# Cálculo da evolução dos casos (ADICIONE ESTA PARTE)
df_evolucao = df_filtrado[df_filtrado['classificacaoFinal'] != 'Descartado']
evolucao_counts = df_evolucao.groupby(['classificacaoFinal', 'evolucaoCaso']).size().reset_index(name='count')
evolucao_counts['percent'] = evolucao_counts.groupby('classificacaoFinal')['count'].transform(lambda x: 100 * x / x.sum())
evolucao_counts['adjusted_percent'] = np.log1p(evolucao_counts['percent'])
evolucao_counts['adjusted_percent'] = evolucao_counts['adjusted_percent'] / evolucao_counts['adjusted_percent'].max() * 100

# -----------------------------------------------------------------------------
# Layout das Abas Atualizado
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Panorama Geral", 
    "👥 Sintomas e Comorbidades",  # Nome corrigido
    "🕒 Análise Temporal",
    "🔮 Predição de Casos"
])

# -----------------------------------------------------------------------------
# Aba 1: Visão Geral
# -----------------------------------------------------------------------------
with tab1:
    st.header("Indicadores Chave e Distribuições")
    
    # Linha 1: KPIs
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

    # Linha 2: Gráficos Principais
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Distribuição de Casos
        fig1 = px.pie(
            df_filtrado,
            names='classificacaoFinal',
            title='Distribuição de Casos por Classificação Final',
            color='classificacaoFinal',
            color_discrete_map=color_map,
            labels={'classificacaoFinal': 'Classificação do Caso'}
        )
        fig1 = apply_chart_styling(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico de Status de Vacinação
        st.subheader("Distribuição Vacinal por Classificação")
        fig_vacina = px.histogram(
            df_filtrado,
            x='status_vacinacao',
            color='classificacaoFinal',
            barmode='group',
            title='Status Vacinal dos Casos',
            labels={'status_vacinacao': 'Status Vacinal'},
            category_orders={"status_vacinacao": ["Sim", "Não", "Ignorado", "Não informado"]},
            color_discrete_map=color_map
        )
        fig_vacina = apply_chart_styling(fig_vacina)
        st.plotly_chart(fig_vacina, use_container_width=True)

    with col2:
        # Gráfico de Evolução dos Casos (lado a lado por classificaçãoFinal)
        fig2 = px.bar(
            evolucao_counts,
            x='classificacaoFinal',
            y='adjusted_percent',
            color='evolucaoCaso',
            barmode='group',  # Agrupamento lado a lado
            title='Evolução Clínica dos Casos',
            labels={
                'classificacaoFinal': 'Classificação',
                'adjusted_percent': 'Porcentagem de Casos',
                'evolucaoCaso': 'Evolução'
            },
            color_discrete_map=color_map  # Se você estiver usando um mapa de cores customizado
        )

        fig2 = apply_chart_styling(fig2)
        st.plotly_chart(fig2, use_container_width=True)


        # Gráficos Demográficos
        st.subheader("Perfil Demográfico")
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            # Distribuição por Sexo
            fig_sexo = px.pie(
                sexo_counts,
                values='Contagem',
                names='Sexo',
                title='Distribuição por Sexo',
                color_discrete_sequence=[descartados_color, confirmados_color]  # Cores invertidas
            )
            fig_sexo = apply_chart_styling(fig_sexo)
            st.plotly_chart(fig_sexo, use_container_width=True)
        
        with demo_col2:
            # Profissionais de Saúde
            fig_prof = px.pie(
                prof_counts,
                values='Contagem',
                names='Categoria',
                title='Profissionais de Saúde Afetados',
                color_discrete_sequence=[descartados_color, confirmados_color]
            )
            fig_prof = apply_chart_styling(fig_prof)
            st.plotly_chart(fig_prof, use_container_width=True)

    # Linha 3: Gráficos Adicionais
    col3, col4 = st.columns(2)
    
    with col3:
        # Distribuição Etária
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
        fig3 = apply_chart_styling(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Status de Vacinação Detalhado
        st.subheader("Status de Vacinação na População")
        fig6 = px.histogram(
            df_filtrado,
            x='status_vacinacao',
            color='classificacaoFinal',
            barmode='group',
            title='Cobertura Vacinal por Classificação',
            category_orders={"status_vacinacao": ["Sim", "Não", "Ignorado", "Não informado"]},
            color_discrete_map=color_map
        )
        fig6.update_layout(legend_title_text="Filtrar")
        fig6 = apply_chart_styling(fig6)
        st.plotly_chart(fig6, use_container_width=True)

# -----------------------------------------------------------------------------
# Aba 3: Sintomas e Condições Pré-existentes
# -----------------------------------------------------------------------------
with tab2:
    st.header("Sintomas e Condições Pré-existentes")
    sintomas = ['Dor de Garganta', 'Coriza', 'Dispneia', 'Distúrbios Olfativos',
                'Distúrbios Gustativos', 'Febre', 'Tosse', 'Dor de Cabeça']
    st.subheader("Frequência de Sintomas")
    sintoma_counts = df_filtrado[sintomas].sum().reset_index()
    sintoma_counts.columns = ['Sintoma', 'Contagem']
    fig7 = px.bar(
        sintoma_counts,
        x='Contagem',
        y='Sintoma',
        orientation='h',
        title='Sintomas Mais Comuns'
    )
    fig7.update_layout(legend_title_text="Filtrar")
    fig7 = apply_chart_styling(fig7)
    st.plotly_chart(fig7, use_container_width=True)
    
    st.subheader("Condições Pré-existentes")
    df_filtrado['condicoes'] = df_filtrado['condicoes'].str.strip("[]").str.replace("'", "")
    df_condicoes = df_filtrado['condicoes'].str.get_dummies(sep=', ')
    if not df_condicoes.empty:
        condicao_counts = df_condicoes.sum().sort_values(ascending=False).reset_index()
        condicao_counts.columns = ['Condição', 'Contagem']
        fig8 = px.bar(
            condicao_counts,
            x='Contagem',
            y='Condição',
            orientation='h',
            title='Condições Pré-existentes Mais Comuns'
        )
        fig8.update_layout(legend_title_text="Filtrar")
        fig8 = apply_chart_styling(fig8)
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.warning("Não há dados suficientes sobre condições pré-existentes.")

# -----------------------------------------------------------------------------
# Aba 3: Análise Temporal
# -----------------------------------------------------------------------------
with tab3:
    st.header("Análise Temporal")
    
    # Gráfico 1: Casos por Data de Notificação (2022-2023)
    st.subheader("Casos por Mês de Notificação (2022-2023)")
    if not df_filtrado['dataNotificacao'].isnull().all():
        # Filtrar período
        mask = (df_filtrado['dataNotificacao'] >= '2022-01-01') & (df_filtrado['dataNotificacao'] <= '2023-12-31')
        df_temporal = df_filtrado[mask].copy()
        
        # Agrupar por mês com período completo
        df_temporal['mes_notificacao'] = df_temporal['dataNotificacao'].dt.to_period('M')
        counts = df_temporal.groupby('mes_notificacao').size().reset_index(name='Contagem')
        counts['mes_notificacao'] = counts['mes_notificacao'].dt.to_timestamp()
        
        # Criar range completo de meses
        all_months = pd.date_range(start='2022-01-01', end='2023-12-31', freq='MS')
        counts = counts.set_index('mes_notificacao').reindex(all_months).fillna(0).reset_index()
        counts.columns = ['Mês', 'Casos']

        fig9 = px.line(
            counts,
            x='Mês',
            y='Casos',
            title='Evolução Mensal de Notificações (2022-2023)',
            markers=True,
            line_shape='spline'
        )
        fig9.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=counts['Mês'],
                tickformat="%b/%Y",
                dtick="M1"
            )
        )
        fig9 = apply_chart_styling(fig9)
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.warning("Dados de data de notificação ausentes ou inválidos.")

    # Gráfico 2: Tempo entre Sintomas e Notificação (Modificado)
    st.subheader("Tempo Médio de Notificação por Classificação")
    if 'dias_entre_sintomas_notificacao' in df_filtrado.columns:
        # Filtrar dados
        df_tempo = df_filtrado[
            (df_filtrado['dias_entre_sintomas_notificacao'] <= 365) &
            (df_filtrado['classificacaoFinal'] != 'Descartado')
        ].copy()
        
        # Calcular médias
        tempo_medio = df_tempo.groupby(['classificacaoFinal', 'evolucaoCaso'])['dias_entre_sintomas_notificacao']\
            .mean().reset_index(name='Dias')
        
        fig10 = px.bar(
            tempo_medio,
            x='classificacaoFinal',
            y='Dias',
            color='evolucaoCaso',
            barmode='group',
            title='Tempo Médio entre Sintomas e Notificação',
            labels={
                'classificacaoFinal': 'Classificação do Caso',
                'Dias': 'Dias Médios',
                'evolucaoCaso': 'Evolução'
            },
            text_auto='.1f'
        )
        fig10.update_traces(textfont_size=12, textangle=0)
        fig10 = apply_chart_styling(fig10)
        st.plotly_chart(fig10, use_container_width=True)
    else:
        st.warning("Dados sobre tempo entre sintomas e notificação não disponíveis.")

# -----------------------------------------------------------------------------
# Aba 5: Previsão de Casos
# -----------------------------------------------------------------------------
with tab4:
    st.header("Previsão de Casos de COVID-19")
    st.markdown("Informe os dados do paciente para realizar a previsão")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            idade = st.number_input("Idade", min_value=0, max_value=120, value=30)
            sexo = st.radio("Sexo", ['Masculino', 'Feminino', 'Indefinido'])
            profissional_saude = st.radio("Profissional de Saúde?", ['Sim', 'Não'])
            vacinado = st.radio("Recebeu vacina?", ['Sim', 'Não', 'Ignorado'])
            total_condicoes = st.number_input("Número de condições pré-existentes", min_value=0, max_value=10, value=0)
        with col2:
            dias_sintomas_notificacao = st.number_input("Dias entre sintomas e notificação", min_value=0, max_value=30, value=3)
            febre = st.checkbox("Febre")
            tosse = st.checkbox("Tosse")
            dor_garganta = st.checkbox("Dor de Garganta")
            dispneia = st.checkbox("Dificuldade Respiratória (Dispneia)")
            disturbios_olfativos = st.checkbox("Distúrbios Olfativos")
            disturbios_gustativos = st.checkbox("Distúrbios Gustativos")
            dor_cabeca = st.checkbox("Dor de Cabeça")
            coriza = st.checkbox("Coriza")
            outros = st.checkbox("Outros Sintomas")
            assintomatico = st.checkbox("Assintomático")
        submitted = st.form_submit_button("Realizar Previsão")
    if submitted:
        try:
            model = joblib.load('melhor_modelo.pkl')
            scaler = joblib.load('scaler.pkl')
            features = [
                idade,
                1 if profissional_saude == 'Não' else 0,
                1 if profissional_saude == 'Sim' else 0,
                1 if sexo == 'Feminino' else 0,
                1 if sexo == 'Indefinido' else 0,
                1 if sexo == 'Masculino' else 0,
                1 if vacinado == 'Ignorado' else 0,
                1 if vacinado == 'Não' else 0,
                1 if vacinado == 'Sim' else 0,
                total_condicoes,
                dias_sintomas_notificacao,
                dor_garganta,
                coriza,
                outros,
                dispneia,
                disturbios_olfativos,
                assintomatico,
                disturbios_gustativos,
                febre,
                tosse,
                dor_cabeca
            ]
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            proba = model.predict_proba(features_scaled)
            st.subheader("Resultado da Previsão:")
            classe_predita = prediction[0]
            confianca = np.max(proba[0]) * 100
            if classe_predita == 'Confirmado':
                st.error(f"**Resultado:** {classe_predita} (Confiança: {confianca:.1f}%)")
            else:
                st.success(f"**Resultado:** {classe_predita} (Confiança: {confianca:.1f}%)")
            st.markdown("### Fatores mais relevantes para a decisão:")
            importances = model.feature_importances_
            feature_names = [
                'Idade', 'Profissional Saúde Não', 'Profissional Saúde Sim', 
                'Sexo Feminino', 'Sexo Indefinido', 'Sexo Masculino', 
                'Vacina Ignorado', 'Vacina Não', 'Vacina Sim', 'Total Condições',
                'Dias Sintomas/Notificação', 'Dor Garganta', 'Coriza', 'Outros',
                'Dispneia', 'Distúrbios Olfativos', 'Assintomático', 
                'Distúrbios Gustativos', 'Febre', 'Tosse', 'Dor Cabeça'
            ]
            indices = np.argsort(importances)[::-1][:5]
            st.write("Principais fatores que influenciaram a previsão:")
            for i in indices:
                st.write(f"- {feature_names[i]} ({importances[i]*100:.1f}%)")
        except Exception as e:
            st.error(f"Erro na previsão: {str(e)}")

# -----------------------------------------------------------------------------
# Rodapé
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("Dashboard desenvolvido para análise de dados de COVID-19 e síndromes gripais")
