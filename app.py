import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Dashboard COVID-19", page_icon="🦠", layout="wide")

# Título do dashboard
st.title("Análise de Casos de COVID-19 e Síndromes Gripais")
st.markdown("Dashboard interativo para análise de casos de COVID-19 e síndromes gripais")

# Carregar os dados
@st.cache_data
def load_data():
    return pd.read_csv("df_preprocessed_new.csv")

df = load_data()

# Converter datas
date_cols = ['dataNotificacao', 'dataInicioSintomas', 'dataEncerramento']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Sidebar com filtros
st.sidebar.header("Filtros")
with st.sidebar:
    # Filtro por classificação
    classificacoes = st.multiselect(
        "Classificação",
        options=df['classificacaoFinal'].unique(),
        default=df['classificacaoFinal'].unique()
    )
    
    # Filtro por evolução
    evolucoes = st.multiselect(
        "Evolução do Caso",
        options=df['evolucaoCaso'].unique(),
        default=df['evolucaoCaso'].unique()
    )
    
    # Filtro por idade
    idade_min, idade_max = st.slider(
        "Faixa Etária",
        min_value=int(df['idade'].min()),
        max_value=int(df['idade'].max()),
        value=(int(df['idade'].min()), int(df['idade'].max()))
    )

# Aplicar filtros
df_filtrado = df[
    (df['classificacaoFinal'].isin(classificacoes)) &
    (df['evolucaoCaso'].isin(evolucoes)) &
    (df['idade'] >= idade_min) &
    (df['idade'] <= idade_max)
]

# Layout do dashboard
tab1, tab2, tab3, tab4 = st.tabs([
    "Visão Geral", 
    "Distribuição Demográfica", 
    "Sintomas e Comorbidades",
    "Temporalidade"
])

with tab1:
    st.header("Visão Geral dos Casos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico 1: Distribuição de casos
        fig1 = px.pie(
            df_filtrado,
            names='classificacaoFinal',
            title='Distribuição de Casos por Classificação'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # Gráfico 2: Evolução dos casos
        fig2 = px.histogram(
            df_filtrado,
            x='evolucaoCaso',
            color='classificacaoFinal',
            barmode='group',
            title='Evolução dos Casos por Classificação'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Distribuição por idade
    st.subheader("Distribuição por Idade")
    fig3 = px.histogram(
        df_filtrado,
        x='idade',
        nbins=20,
        color='classificacaoFinal',
        title='Distribuição de Casos por Idade'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("Análise Demográfica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico 4: Distribuição por sexo
        sexo_counts = df_filtrado[[
            'sexo_Feminino', 
            'sexo_Masculino', 
            'sexo_Indefinido'
        ]].sum().reset_index()
        sexo_counts.columns = ['Sexo', 'Contagem']
        sexo_counts['Sexo'] = sexo_counts['Sexo'].str.replace('sexo_', '')
        
        fig4 = px.pie(
            sexo_counts,
            values='Contagem',
            names='Sexo',
            title='Distribuição por Sexo'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # Gráfico 5: Profissionais de saúde
        prof_counts = df_filtrado[[
            'profissionalSaude_Sim', 
            'profissionalSaude_Não'
        ]].sum().reset_index()
        prof_counts.columns = ['Categoria', 'Contagem']
        prof_counts['Categoria'] = prof_counts['Categoria'].str.replace('profissionalSaude_', '')
        
        fig5 = px.bar(
            prof_counts,
            x='Categoria',
            y='Contagem',
            title='Profissionais de Saúde Afetados'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Gráfico 6: Vacinação
    st.subheader("Status de Vacinação")
    df_filtrado['status_vacinacao'] = 'Não informado'
    df_filtrado.loc[df_filtrado['codigoRecebeuVacina_Sim'] == 1, 'status_vacinacao'] = 'Sim'
    df_filtrado.loc[df_filtrado['codigoRecebeuVacina_Não'] == 1, 'status_vacinacao'] = 'Não'
    df_filtrado.loc[df_filtrado['codigoRecebeuVacina_Ignorado'] == 1, 'status_vacinacao'] = 'Ignorado'

    fig6 = px.histogram(
        df_filtrado,
        x='status_vacinacao',
        color='classificacaoFinal',
        barmode='group',
        title='Distribuição por Status de Vacinação',
        category_orders={"status_vacinacao": ["Sim", "Não", "Ignorado", "Não informado"]}
    )
    st.plotly_chart(fig6, use_container_width=True)

with tab3:
    st.header("Sintomas e Condições Pré-existentes")
    
    # Lista de sintomas
    sintomas = [
        'Dor de Garganta', 'Coriza', 'Dispneia', 'Distúrbios Olfativos',
        'Distúrbios Gustativos', 'Febre', 'Tosse', 'Dor de Cabeça'
    ]
    
    # Gráfico 7: Frequência de sintomas
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
    st.plotly_chart(fig7, use_container_width=True)
    
    # Gráfico 8: Condições pré-existentes
    st.subheader("Condições Pré-existentes")
    
    # Extrair condições (assumindo que a coluna 'condicoes' contém strings como "['Diabetes', 'Gestante']")
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
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.warning("Não há dados suficientes sobre condições pré-existentes.")

with tab4:
    st.header("Análise Temporal")
    
    # Gráfico 9: Casos ao longo do tempo
    st.subheader("Casos por Data de Notificação")
    
    if not df_filtrado['dataNotificacao'].isnull().all():
        df_temporal = df_filtrado.groupby(
            df_filtrado['dataNotificacao'].dt.to_period('M')
        ).size().reset_index(name='Contagem')
        df_temporal['dataNotificacao'] = df_temporal['dataNotificacao'].dt.to_timestamp()
        
        fig9 = px.line(
            df_temporal,
            x='dataNotificacao',
            y='Contagem',
            title='Casos por Mês de Notificação'
        )
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.warning("Dados de data de notificação ausentes ou inválidos.")
    
    # Gráfico 10: Tempo entre sintomas e notificação
    st.subheader("Tempo entre Sintomas e Notificação")
    
    if 'dias_entre_sintomas_notificacao' in df_filtrado.columns:
        fig10 = px.box(
            df_filtrado,
            x='classificacaoFinal',
            y='dias_entre_sintomas_notificacao',
            color='evolucaoCaso',
            title='Tempo entre Início dos Sintomas e Notificação'
        )
        st.plotly_chart(fig10, use_container_width=True)
    else:
        st.warning("Dados sobre tempo entre sintomas e notificação não disponíveis.")

# Rodapé
st.markdown("---")
st.markdown("Dashboard desenvolvido para análise de dados de COVID-19 e síndromes gripais")