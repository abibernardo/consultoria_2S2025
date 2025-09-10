
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Caminho do arquivo Excel
caminho = "https://raw.githubusercontent.com/abibernardo/consultoria_2S2025/main/C%C3%B3pia%20de%20Resultados%20Finais%20-%20Projeto%20Congelamento%20JBS.xlsx"

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
#corr_focus = corr.loc[respostas, num_cols]

# Cria heatmap
heatmap = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=corr.round(2).values,
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

### MANOVA

# Variáveis dependentes
y = df[["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]]

# Fator independente
x = df["GRUPO"]

# Montando a fórmula: "var1 + var2 + var3 ~ GRUPO"
formula = "Q('MÉDIA a*') + Q('MÉDIA b*') + Q('MÉDIA L*') ~ GRUPO"

manova = MANOVA.from_formula(formula, data=df)
# Resultados da MANOVA
resultado = manova.mv_test()

st.subheader("📊 MANOVA - Teste Multivariado")
st.text(str(resultado))

## ANOVAS univariadas


st.subheader("📑 ANOVAs univariadas (post-hoc)")
for var in ["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]:
    modelo = ols(f"Q('{var}') ~ GRUPO", data=df).fit()
    anova_res = anova_lm(modelo)
    st.text(f"ANOVA para {var}:")
    st.dataframe(anova_res.round(3))
