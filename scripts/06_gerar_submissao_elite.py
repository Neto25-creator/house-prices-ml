"""
Etapa 6 (Fase 2 - Modelo Elite): Geração do Arquivo Final submission_elite.csv
------------------------------------------------------------------------------
Objetivo: Utilizar o comitê de 5 modelos de elite calibrado na Etapa 5
para gerar as previsões finais de teste e salvar o arquivo 'submission_elite.csv'.

Comitê de Ponderação:
 - 25% XGBoost Elite
 - 25% Gradient Boosting Huber
 - 20% Lasso com RobustScaler
 - 15% ElasticNet com RobustScaler
 - 15% Ridge com RobustScaler
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DADOS_ELITE = BASE_DIR / "dados" / "dados_tratados_elite"
DIR_MODELOS_ELITE = BASE_DIR / "modelos_salvos_elite"
ARQUIVO_SUBMISSAO_ELITE = BASE_DIR / "submission_elite.csv"


def main():
    print("=" * 80)
    print(" ETAPA 6 (FASE 2 - MODELO ELITE): GERAÇÃO DO ARQUIVO SUBMISSION_ELITE.CSV")
    print("=" * 80)

    # 1. Carregamento dos dados de teste de elite
    caminho_teste = DIR_DADOS_ELITE / "test_elite.csv"
    print(f"\n[1] Carregando '{caminho_teste}'...")
    df_teste = pd.read_csv(caminho_teste)
    ids = df_teste["Id"]
    X_teste = df_teste.drop(columns=["Id"])

    print(f" -> Imóveis a prever: {len(ids)} (de Id {ids.min()} até {ids.max()})")
    print(f" -> Variáveis preditoras: {X_teste.shape[1]} colunas")

    # 2. Carregando os 5 modelos de elite
    print(f"\n[2] Carregando os 5 modelos de '{DIR_MODELOS_ELITE}'...")
    m_xgb = joblib.load(DIR_MODELOS_ELITE / "modelo_xgb_elite.pkl")
    m_gbr = joblib.load(DIR_MODELOS_ELITE / "modelo_gbr_elite.pkl")
    m_lasso = joblib.load(DIR_MODELOS_ELITE / "modelo_lasso_elite.pkl")
    m_enet = joblib.load(DIR_MODELOS_ELITE / "modelo_enet_elite.pkl")
    m_ridge = joblib.load(DIR_MODELOS_ELITE / "modelo_ridge_elite.pkl")
    print(" -> XGBoost, GBR Huber, Lasso, ElasticNet e Ridge carregados!")

    # 3. Gerando previsões individuais
    print("\n[3] Calculando previsões para cada modelo...")
    p_xgb = m_xgb.predict(X_teste)
    p_gbr = m_gbr.predict(X_teste)
    p_lasso = m_lasso.predict(X_teste)
    p_enet = m_enet.predict(X_teste)
    p_ridge = m_ridge.predict(X_teste)

    # 4. Combinação do Comitê de Elite
    print("\n[4] Aplicando ponderação do Comitê de Elite...")
    print(" -> 25% XGBoost + 25% GBR + 20% Lasso + 15% ElasticNet + 15% Ridge")
    p_blend_log = (
        0.25 * p_xgb
        + 0.25 * p_gbr
        + 0.20 * p_lasso
        + 0.15 * p_enet
        + 0.15 * p_ridge
    )

    # 5. Transformação Inversa (expm1) para dólares reais
    print("\n[5] Convertendo previsões para Dólares reais (np.expm1)...")
    precos_finais = np.expm1(p_blend_log)

    # 6. DataFrame de Submissão
    df_submissao = pd.DataFrame({"Id": ids, "SalePrice": precos_finais})

    # 7. Testes de Sanidade e Conformidade
    print("\n" + "-" * 80)
    print(" [6] TESTES DE SANIDADE E CONFORMIDADE COM O KAGGLE")
    print("-" * 80)
    assert df_submissao.shape == (
        1459,
        2,
    ), f"Formato inválido: {df_submissao.shape}"
    print(" [OK] Formato exato: 1.459 linhas e 2 colunas ('Id' e 'SalePrice')")

    nulos = df_submissao.isnull().sum().sum()
    assert nulos == 0, f"Erro: Existem {nulos} valores nulos!"
    print(" [OK] Valores nulos ou ausentes: 0")

    menor_p = df_submissao["SalePrice"].min()
    assert menor_p > 0, f"Preço inválido: {menor_p}"
    print(f" [OK] Menor preço previsto : US$ {menor_p:>12,.2f}")
    print(
        f" [OK] Preço médio previsto : US$ {df_submissao['SalePrice'].mean():>12,.2f}"
    )
    print(
        f" [OK] Maior preço previsto : US$ {df_submissao['SalePrice'].max():>12,.2f}"
    )

    # 8. Salvando arquivo final de elite
    df_submissao.to_csv(ARQUIVO_SUBMISSAO_ELITE, index=False)
    print("\n" + "=" * 80)
    print(
        f" ARQUIVO GERADO COM SUCESSO: '{ARQUIVO_SUBMISSAO_ELITE.name}' (na raiz do projeto)"
    )
    print("=" * 80)

    print("\nPrimeiras 5 previsões de Elite:")
    print(df_submissao.head().to_string(index=False))

    print("\nÚltimas 5 previsões de Elite:")
    print(df_submissao.tail().to_string(index=False))


if __name__ == "__main__":
    main()
