# Configuração da página
st.set_page_config(page_title="Análise - Alcatra", layout="wide")

st.title("📊 Análise Exploratória - Variáveis de Cor (MÉDIAS)")

# Seleção das colunas de interesse
variaveis = ["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]

# ---------------- Estatísticas descritivas ----------------
st.subheader("📑 Estatísticas por Grupo")

estatisticas = df.groupby("GRUPO")[variaveis].agg(
    ["mean", "std", "var"]
).round(2)

st.dataframe(estatisticas, use_container_width=True)

# ---------------- Boxplots ----------------
st.subheader("Distribuição por Grupo (Boxplots)")
for var in variaveis:
    fig = px.box(df, x="GRUPO", y=var, color="GRUPO",
                 title=f"Distribuição de {var} por Grupo",
                 points="all")
    st.plotly_chart(fig, use_container_width=True)

st.plotly_chart(fig, use_container_width=True)
