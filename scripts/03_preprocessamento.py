"""
Etapa 3 (Fase 1 - Versão 0.1292): Pré-processamento e Engenharia de Atributos
-----------------------------------------------------------------------------
Este script realiza o pipeline original da Fase 1 (RMSLE = 0.1292):
 1. Trata 100% dos valores ausentes (conforme validado na Etapa 2).
 2. Cria variáveis combinadas fundamentais (TotalSF, TotalBath, etc.).
 3. Converte notas de qualidade e condição em números ordinais (1 a 5).
 4. Transforma categorias de texto em colunas numéricas (One-Hot Encoding).
 5. Aplica a transformação logarítmica na variável alvo (SalePrice).
 6. Salva os dados em 'dados/dados_tratados/train_processado.csv' e 'test_processado.csv'.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DADOS_INICIAIS = BASE_DIR / "dados" / "dados_iniciais"
DIR_DADOS_TRATADOS = BASE_DIR / "dados" / "dados_tratados"


def processar_dados():
    print("=" * 70)
    print(" ETAPA 3 (FASE 1): PRÉ-PROCESSAMENTO ORIGINAL (VERSÃO 0.1292)")
    print("=" * 70)

    # 1. Carregamento dos dados brutos
    caminho_train = DIR_DADOS_INICIAIS / "train.csv"
    caminho_test = DIR_DADOS_INICIAIS / "test.csv"
    print(f"\n[1] Carregando arquivos brutos de '{DIR_DADOS_INICIAIS}'...")
    df_train = pd.read_csv(caminho_train)
    df_test = pd.read_csv(caminho_test)

    ids_teste = df_test["Id"]
    precos_venda = df_train["SalePrice"]

    # 2. Tratamento dos Valores Ausentes
    print("\n[2] Tratando valores ausentes...")

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

    # Grupo B: Imputação sem vazamento de dados
    medianas_frente_bairro = df_train.groupby("Neighborhood")[
        "LotFrontage"
    ].median()
    df_train["LotFrontage"] = df_train["LotFrontage"].fillna(
        df_train["Neighborhood"].map(medianas_frente_bairro)
    )
    df_test["LotFrontage"] = df_test["LotFrontage"].fillna(
        df_test["Neighborhood"].map(medianas_frente_bairro)
    )

    cols_moda = [
        "MSZoning",
        "Utilities",
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

    print(" -> Ausentes preenchidos com sucesso em Treino e Teste!")

    # 3. Engenharia de Atributos da Fase 1
    print("\n[3] Criando variáveis preditivas da Fase 1...")
    for df in [df_train, df_test]:
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
        df["IdadeImovel"] = df["YrSold"] - df["YearBuilt"]
        df["IdadeReforma"] = df["YrSold"] - df["YearRemodAdd"]
        df["TemGaragem"] = (df["GarageArea"] > 0).astype(int)
        df["TemPorao"] = (df["TotalBsmtSF"] > 0).astype(int)
        df["TemPiscina"] = (df["PoolArea"] > 0).astype(int)

    # 4. Codificação Ordinal
    print("\n[4] Convertendo escalas ordinais de qualidade em notas (1 a 5)...")
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

    # 5. One-Hot Encoding
    print("\n[5] Aplicando One-Hot Encoding nas variáveis nominais restantes...")
    X_train_bruto = df_train.drop(columns=["Id", "SalePrice"])
    X_test_bruto = df_test.drop(columns=["Id"])

    X_train_encoded = pd.get_dummies(X_train_bruto, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_bruto, drop_first=True)

    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns, fill_value=0
    )

    X_train_encoded = X_train_encoded.astype(int, errors="ignore")
    X_test_encoded = X_test_encoded.astype(int, errors="ignore")

    # 6. Transformação Logarítmica do Alvo
    y_train_log = np.log1p(precos_venda)

    # 7. Salvando arquivos processados da Fase 1
    print(
        f"\n[6] Exportando arquivos CSV pré-processados da Fase 1 para '{DIR_DADOS_TRATADOS}'..."
    )
    DIR_DADOS_TRATADOS.mkdir(parents=True, exist_ok=True)

    df_train_pronto = X_train_encoded.copy()
    df_train_pronto["SalePrice"] = precos_venda
    df_train_pronto["SalePrice_Log"] = y_train_log

    df_test_pronto = X_test_encoded.copy()
    df_test_pronto["Id"] = ids_teste

    caminho_salvar_train = DIR_DADOS_TRATADOS / "train_processado.csv"
    caminho_salvar_test = DIR_DADOS_TRATADOS / "test_processado.csv"

    df_train_pronto.to_csv(caminho_salvar_train, index=False)
    df_test_pronto.to_csv(caminho_salvar_test, index=False)

    print(
        f" -> '{caminho_salvar_train.name}' salvo: {df_train_pronto.shape[0]} linhas e {df_train_pronto.shape[1]} colunas"
    )
    print(
        f" -> '{caminho_salvar_test.name}'  salvo: {df_test_pronto.shape[0]} linhas e {df_test_pronto.shape[1]} colunas"
    )

    print("\n" + "=" * 70)
    print(" SUCESSO: Dados originais da Fase 1 (1.460 imóveis) restaurados!")
    print("=" * 70)

    return (
        X_train_encoded,
        precos_venda,
        y_train_log,
        X_test_encoded,
        ids_teste,
    )


if __name__ == "__main__":
    processar_dados()
