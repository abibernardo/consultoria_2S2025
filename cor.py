
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Caminho do arquivo Excel
caminho = "https://raw.githubusercontent.com/abibernardo/consultoria_2S2025/main/C%C3%B3pia%20de%20Resultados%20Finais%20-%20Projeto%20Congelamento%20JBS.xlsx"

# Leitura da planilha "Alcatra"
df = pd.read_excel(caminho, sheet_name="Alcatra")
#df.rename(columns={'MATURAÇÃO': 'MATURACAO'}, inplace=True)

# Configuração da página
st.set_page_config(page_title="Análise de coloração - Alcatra", layout="wide")

option = st.selectbox(
    " ",
    ("Análise Exploratória", "Modelagem"),
)

if option == 'Análise Exploratória':

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
        fig = px.box(
            df,
            x="MATURAÇÃO ",
            y=var,
            color="GRUPO",
            facet_col="GRUPO",
            title=f"Distribuição de {var} por Maturação e Grupo",
            points="all"
        )
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

elif option == 'Modelagem':

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

    def parse_manova_table(mv_test_res):
        tables = []
        for effect, res in mv_test_res.results.items():  # <- CORREÇÃO AQUI
            stat_table = res["stat"]
            for test in stat_table.index:
                row = stat_table.loc[test]
                tables.append({
                    "Efeito": effect,
                    "Estatística": test,
                    "Valor": row["Value"],
                    "Num DF": row["Num DF"],
                    "Den DF": row["Den DF"],
                    "F": row["F Value"],
                    "p-valor": row["Pr > F"]
                })
        return pd.DataFrame(tables)

    manova_df = parse_manova_table(resultado)
    manova_df["p-valor"] = manova_df["p-valor"].apply(lambda x: "<0.0001" if x == 0 else round(x, 4))


    st.subheader("📊 Resultados da MANOVA (formatados)")
    st.dataframe(manova_df.round(4), use_container_width=True)


    ## Testando pressupostos

    def box_m_test(data, group):
        """
        Teste M de Box para homogeneidade das matrizes de covariância.

        data: DataFrame apenas com variáveis dependentes (contínuas)
        group: Series ou coluna categórica com os grupos
        """
        groups = group.unique()
        n_groups = len(groups)
        n_vars = data.shape[1]
        n_total = data.shape[0]

        # Matrizes de covariância dentro de cada grupo
        cov_matrices = []
        ns = []
        for g in groups:
            subset = data[group == g]
            ns.append(len(subset))
            cov_matrices.append(np.cov(subset.T, bias=False))

        # Pooled covariance matrix
        pooled_cov = sum([(ns[i] - 1) * cov_matrices[i] for i in range(n_groups)]) / (n_total - n_groups)

        # Determinantes
        logdet_pooled = np.log(np.linalg.det(pooled_cov))
        logdet_group = np.sum([(ns[i] - 1) * np.log(np.linalg.det(cov_matrices[i])) for i in range(n_groups)])

        # Estatística de teste
        C = ((n_total - n_groups) * logdet_pooled - logdet_group)
        m = ((2 * n_vars ** 2 + 3 * n_vars - 1) / (6 * (n_vars + 1) * (n_groups - 1))) * (
                sum([1 / (ns[i] - 1) for i in range(n_groups)]) - 1 / (n_total - n_groups)
        )
        chi_stat = (1 - m) * C
        df = (n_groups - 1) * n_vars * (n_vars + 1) / 2
        p_value = 1 - chi2.cdf(chi_stat, df)

        return chi_stat, df, p_value


    # Aplicação aos seus dados
    Y = df[["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]]
    grupo = df["GRUPO"]

    chi_stat, df_box, p_value = box_m_test(Y, grupo)

    st.subheader("📊 Teste M de Box")
    st.write(f"Estatística Qui-quadrado: {chi_stat:.3f}")
    st.write(f"Graus de liberdade: {df_box:.0f}")
    st.write(f"p-valor: {p_value:.5f}")

    ## ANOVAS univariadas


    st.subheader("📑 ANOVAs univariadas")
    for var in ["MÉDIA a*", "MÉDIA b*", "MÉDIA L*"]:
        modelo = ols(f"Q('{var}') ~ GRUPO", data=df).fit()
        anova_res = anova_lm(modelo)
        st.text(f"ANOVA para {var}:")
        st.dataframe(anova_res.round(3))



