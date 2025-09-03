import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.stattools import pacf
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import pandas as pd
import plotly.figure_factory as ff

import streamlit as st
import pandas as pd

# Caminho do arquivo Excel
caminho = r"C:\Users\Usuário\Downloads\Cópia de Resultados Finais - Projeto Congelamento JBS.xlsx"

# Leitura da planilha "Alcatra"
df = pd.read_excel(caminho, sheet_name="Alcatra")
#df.rename(columns={'MATURAÇÃO': 'MATURACAO'}, inplace=True)

# Configuração da página
st.set_page_config(page_title="Análise - Alcatra", layout="wide")

st.title("📊 Análise Exploratória - Variáveis de Cor (MÉDIAS)")
st.subheader(f"Em cada análise, selecione parte dos gráficos para dar zoom e abstrair os outliars !")
st.divider()
#st.dataframe(df)


# Seleção das colunas de interesse
variaveis = ["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]

# ---------------- Estatísticas descritivas ----------------
st.subheader("📑 Estatísticas por Grupo")

estatisticas = df.groupby("GRUPO")[variaveis].agg(
    ["mean", "std", "var"]
).round(2)

st.dataframe(estatisticas, use_container_width=True)

st.divider()

st.title("📊 Distribuições")

# --- Gráfico 3D interativo ---

fig_3d = px.scatter_3d(
    df,
    x="MÉDIA a*",
    y="MÉDIA b*",
    z="MÉDIA L*",
    color="GRUPO",
    symbol="GRUPO",
    opacity=0.8,
    title="Espaço de Cor CIELAB (a*, b*, L*)"
)
fig_3d.update_traces(marker=dict(size=5))
st.plotly_chart(fig_3d, use_container_width=True)


variaveis = ["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]



fig = px.histogram(
    df,
    x='MÉDIA a*',
    color="GRUPO",
    barmode="overlay",
    nbins=50,
    opacity=0.6,
    title=f"Distribuição de a* por Controle e Teste")
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    df,
    x='MÉDIA b*',
    color="GRUPO",
    barmode="overlay",
    nbins=100,
    opacity=0.6,
    title=f"Distribuição de b* por Controle e Teste")
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    df,
    x='MÉDIA L*',
    color="GRUPO",
    barmode="overlay",
    nbins=300,
    opacity=0.6,
    title=f"Distribuição de L* por Controle e Teste")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- Boxplots ----------------
st.subheader("Distribuição por Teste e Controle (Boxplots)")
for var in variaveis:
    fig = px.box(df, x="GRUPO", y=var, color="GRUPO",
                 title=f"Distribuição de {var} por Grupo",
                 points="all")
    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Distribuição por Tempo de Maturação (Boxplots)")
for var in variaveis:
    fig = px.box(df, x="MATURAÇÃO ", y=var, color="MATURAÇÃO ",
                 title=f"Distribuição de {var} por Grupo",
                 points="all")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# --- Heatmap de correlação ---
# Variáveis resposta
respostas = ["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]

# Seleciona todas as numéricas
num_cols = df.select_dtypes(include="number").columns.tolist()
del num_cols[6:15]

# Filtra para pegar só as respostas + todas as outras numéricas
corr = df[respostas + num_cols].corr()

# Pega só as correlações das respostas com as demais variáveis
corr_focus = corr.loc[respostas, num_cols]

# Cria heatmap
heatmap = ff.create_annotated_heatmap(
    z=corr_focus.values,
    x=list(corr_focus.columns),
    y=list(corr_focus.index),
    annotation_text=corr_focus.round(2).values,
    colorscale="RdBu",
    showscale=True,
    reversescale=True
)

st.subheader("Correlação das variáveis resposta (a*, b*, L*) com todas as numéricas")
st.plotly_chart(heatmap, use_container_width=True)

st.divider()

"""st.subheader("Distribuição por Grupo.1 (Útil?)")
for var in variaveis:
    fig = px.box(df, x="GRUPO.1", y=var, color="GRUPO.1",
                 title=f"Distribuição de {var} por Grupo",
                 points="all")
    st.plotly_chart(fig, use_container_width=True)"""
