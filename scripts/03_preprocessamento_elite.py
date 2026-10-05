"""
Etapa 3 (Fase 2 - Modelo Elite): Pré-processamento e Engenharia de Atributos
----------------------------------------------------------------------------
Objetivo: Preparar os dados para romper a barreira de RMSLE <= 0.114
de forma segura, sem data leakage e preservando a separação da Fase 1.

Melhorias implementadas:
 1. Remoção de Outliers Extremos: exclusão dos 2 imóveis anômalos
    (IDs 524 e 1299) documentados cientificamente por Dean De Cock.
 2. Tratamento Semântico Completo de Valores Ausentes.
 3. Engenharia de Recursos Avançada:
    - TotalSF, TotalBath, TotalPorchSF, IdadeImovel, IdadeReforma.
    - Interações de alta relevância: Qualidade_TotalSF e QualidadeTotal.
    - Remoção de variáveis com variância quase zero (Utilities).
 4. Mapeamento Ordinal de notas de qualidade e condição (0 a 5).
 5. Correção de Assimetria (Skewness Correction): aplicação de log1p
    em todas as variáveis numéricas preditoras com |skew| > 0.75.
 6. One-Hot Encoding com alinhamento rigoroso entre Treino e Teste.
 7. Transformação log1p no SalePrice.
 8. Exportação para 'dados/dados_tratados_elite/train_elite.csv' e 'test_elite.csv'.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DADOS_INICIAIS = BASE_DIR / "dados" / "dados_iniciais"
DIR_DADOS_ELITE = BASE_DIR / "dados" / "dados_tratados_elite"


def processar_dados_elite():
    print("=" * 75)
    print(" ETAPA 3 (FASE 2 - MODELO ELITE): PRÉ-PROCESSAMENTO AVANÇADO")
    print("=" * 75)

    # 1. Carregamento dos dados brutos
    caminho_train = DIR_DADOS_INICIAIS / "train.csv"
    caminho_test = DIR_DADOS_INICIAIS / "test.csv"
    print(f"\n[1] Carregando arquivos brutos de '{DIR_DADOS_INICIAIS}'...")
    df_train = pd.read_csv(caminho_train)
    df_test = pd.read_csv(caminho_test)

    # 2. Remoção de Outliers Extremos (Recomendação Dean De Cock)
    print("\n[2] Removendo outliers extremos fundamentados cientificamente...")
    linhas_antes = len(df_train)
    filtro_outliers = (df_train["GrLivArea"] > 4000) & (
        df_train["SalePrice"] < 300000
    )
    ids_removidos = df_train[filtro_outliers]["Id"].tolist()
    df_train = df_train.drop(df_train[filtro_outliers].index).reset_index(
        drop=True
    )
    print(
        f" -> Imóveis anômalos removidos: {ids_removidos} (GrLivArea > 4000 sq ft com preço anômalo)"
    )
    print(
        f" -> Amostras de Treino ajustadas: de {linhas_antes} para {len(df_train)} imóveis"
    )

    ids_teste = df_test["Id"]
    precos_venda = df_train["SalePrice"]

    # 3. Tratamento dos Valores Ausentes
    print("\n[3] Tratando valores ausentes...")

    # Grupo A: Ausência estrutural ("Não possui o item")
    cols_nenhum = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]

    cols_zero = [
        "GarageCars",
        "GarageArea",
        "GarageYrBlt",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ]

    for df in [df_train, df_test]:
        df[cols_nenhum] = df[cols_nenhum].fillna("None")
        df[cols_zero] = df[cols_zero].fillna(0)

    # Grupo B: Imputação sem data leakage (aprendida do treino)
    medianas_frente = df_train.groupby("Neighborhood")["LotFrontage"].median()
    df_train["LotFrontage"] = df_train["LotFrontage"].fillna(
        df_train["Neighborhood"].map(medianas_frente)
    )
    df_test["LotFrontage"] = df_test["LotFrontage"].fillna(
        df_test["Neighborhood"].map(medianas_frente)
    )
    mediana_geral = df_train["LotFrontage"].median()
    df_train["LotFrontage"] = df_train["LotFrontage"].fillna(mediana_geral)
    df_test["LotFrontage"] = df_test["LotFrontage"].fillna(mediana_geral)

    cols_moda = [
        "MSZoning",
        "Exterior1st",
        "Exterior2nd",
        "KitchenQual",
        "Functional",
        "SaleType",
        "Electrical",
    ]
    for col in cols_moda:
        moda_treino = df_train[col].mode()[0]
        df_train[col] = df_train[col].fillna(moda_treino)
        df_test[col] = df_test[col].fillna(moda_treino)

    # Descartando coluna de variância quase nula (ruído estatístico)
    for df in [df_train, df_test]:
        if "Utilities" in df.columns:
            df.drop(columns=["Utilities"], inplace=True)

    print(" -> Ausentes preenchidos com sucesso e 'Utilities' descartada!")

    # 4. Engenharia de Atributos Avançada
    print("\n[4] Criando variáveis e interações preditivas de alto impacto...")
    for df in [df_train, df_test]:
        # Metragens totais
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )
        df["TotalPorchSF"] = (
            df["WoodDeckSF"]
            + df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        )

        # Idades
        df["IdadeImovel"] = df["YrSold"] - df["YearBuilt"]
        df["IdadeReforma"] = df["YrSold"] - df["YearRemodAdd"]
        df["AnoConstrucaoReforma"] = df["YearBuilt"] + df["YearRemodAdd"]

        # Interações de mercado imobiliário
        df["QualidadeTotal"] = df["OverallQual"] * df["OverallCond"]
        df["Qualidade_TotalSF"] = df["OverallQual"] * df["TotalSF"]

        # Indicadores binários
        df["TemGaragem"] = (df["GarageArea"] > 0).astype(int)
        df["TemPorao"] = (df["TotalBsmtSF"] > 0).astype(int)
        df["TemPiscina"] = (df["PoolArea"] > 0).astype(int)

    # 5. Mapeamento Ordinal
    print("\n[5] Convertendo escalas ordinais de qualidade em notas (1 a 5)...")
    mapa_qualidade = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    cols_ordinais = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]
    for col in cols_ordinais:
        df_train[col] = df_train[col].map(mapa_qualidade).fillna(0).astype(int)
        df_test[col] = df_test[col].map(mapa_qualidade).fillna(0).astype(int)

    # 6. Correção de Assimetria (Skewness Correction com log1p)
    print("\n[6] Corrigindo assimetria (skewness) nas variáveis numéricas...")
    cols_numericas = df_train.select_dtypes(include=[np.number]).columns.drop(
        ["Id", "SalePrice"]
    )
    assimetrias = (
        df_train[cols_numericas]
        .apply(lambda x: skew(x.dropna()))
        .sort_values(ascending=False)
    )
    cols_assimetricas = assimetrias[abs(assimetrias) > 0.75].index

    print(
        f" -> {len(cols_assimetricas)} variáveis com |skew| > 0.75 transformadas com log1p"
    )
    for col in cols_assimetricas:
        min_tr = df_train[col].min()
        min_te = df_test[col].min()
        if min_tr >= 0 and min_te >= 0:
            df_train[col] = np.log1p(df_train[col])
            df_test[col] = np.log1p(df_test[col])

    # 7. One-Hot Encoding
    print("\n[7] Aplicando One-Hot Encoding com alinhamento de colunas...")
    X_train_bruto = df_train.drop(columns=["Id", "SalePrice"])
    X_test_bruto = df_test.drop(columns=["Id"])

    X_train_encoded = pd.get_dummies(X_train_bruto, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_bruto, drop_first=True)

    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns, fill_value=0
    )

    X_train_encoded = X_train_encoded.astype(float)
    X_test_encoded = X_test_encoded.astype(float)

    # 8. Transformação Logarítmica do Alvo
    y_train_log = np.log1p(precos_venda)

    # 9. Salvando em dados/dados_tratados_elite/
    print(
        f"\n[8] Exportando arquivos CSV de Elite para '{DIR_DADOS_ELITE}'..."
    )
    DIR_DADOS_ELITE.mkdir(parents=True, exist_ok=True)

    df_train_pronto = X_train_encoded.copy()
    df_train_pronto["SalePrice"] = precos_venda
    df_train_pronto["SalePrice_Log"] = y_train_log

    df_test_pronto = X_test_encoded.copy()
    df_test_pronto["Id"] = ids_teste

    caminho_salvar_train = DIR_DADOS_ELITE / "train_elite.csv"
    caminho_salvar_test = DIR_DADOS_ELITE / "test_elite.csv"

    df_train_pronto.to_csv(caminho_salvar_train, index=False)
    df_test_pronto.to_csv(caminho_salvar_test, index=False)

    print(
        f" -> '{caminho_salvar_train.name}' salvo: {df_train_pronto.shape[0]} linhas e {df_train_pronto.shape[1]} colunas"
    )
    print(
        f" -> '{caminho_salvar_test.name}'  salvo: {df_test_pronto.shape[0]} linhas e {df_test_pronto.shape[1]} colunas"
    )
    print(
        f" -> Nulos restantes em Treino: {df_train_pronto.isnull().sum().sum()}"
    )
    print(
        f" -> Nulos restantes em Teste : {df_test_pronto.isnull().sum().sum()}"
    )

    print("\n" + "=" * 75)
    print(
        " SUCESSO: Dados de Elite (train_elite.csv e test_elite.csv) prontos!"
    )
    print("=" * 75)

    return (
        X_train_encoded,
        precos_venda,
        y_train_log,
        X_test_encoded,
        ids_teste,
    )


if __name__ == "__main__":
    processar_dados_elite()
