"""
Etapa 6: Geração do Arquivo Final de Submissão para o Kaggle
------------------------------------------------------------
Objetivo: Utilizar o comitê de modelos treinado na Etapa 5 para gerar
as previsões de preço para os 1.459 imóveis de teste e salvar o arquivo
'submission.csv' no formato exato exigido pela competição.

Passos realizados:
 1. Carrega os dados de teste pré-processados ('test_processado.csv').
 2. Carrega os 3 modelos treinados da pasta 'modelos_salvos/':
    - XGBoost (50%)
    - Ridge (30%)
    - Random Forest (20%)
 3. Gera as previsões na escala logarítmica para cada modelo.
 4. Combina as previsões no Ensemble ponderado.
 5. Converte os valores de volta para dólares reais usando a função inversa (np.expm1).
 6. Executa testes de sanidade (checagem de nulos, formato, quantidade de linhas e faixas de preço).
 7. Salva o arquivo 'submission.csv'.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DADOS_TRATADOS = BASE_DIR / "dados" / "dados_tratados"
DIR_MODELOS = BASE_DIR / "modelos_salvos"
ARQUIVO_SUBMISSAO = BASE_DIR / "submission.csv"


def main():
    print("=" * 70)
    print(" ETAPA 6: GERAÇÃO DO ARQUIVO FINAL DE SUBMISSÃO (KAGGLE)")
    print("=" * 70)

    # 1. Carregando dados de teste pré-processados
    caminho_teste = DIR_DADOS_TRATADOS / "test_processado.csv"
    print(f"\n[1] Carregando '{caminho_teste}'...")
    df_teste = pd.read_csv(caminho_teste)
    ids = df_teste["Id"]
    X_teste = df_teste.drop(columns=["Id"])

    print(f" -> Imóveis a prever: {len(ids)} (de Id {ids.min()} até {ids.max()})")
    print(f" -> Variáveis preditoras: {X_teste.shape[1]} colunas")

    # 2. Carregando os modelos treinados
    print(f"\n[2] Carregando os modelos treinados de '{DIR_MODELOS}'...")
    modelo_xgb = joblib.load(DIR_MODELOS / "modelo_xgb.pkl")
    modelo_ridge = joblib.load(DIR_MODELOS / "modelo_ridge.pkl")
    modelo_rf = joblib.load(DIR_MODELOS / "modelo_rf.pkl")
    print(" -> XGBoost, Ridge e Random Forest carregados com sucesso!")

    # 3. Gerando previsões individuais (em escala log)
    print("\n[3] Calculando previsões para cada modelo...")
    pred_xgb_log = modelo_xgb.predict(X_teste)
    pred_ridge_log = modelo_ridge.predict(X_teste)
    pred_rf_log = modelo_rf.predict(X_teste)

    # 4. Combinação Ponderada (Ensemble Blending)
    print("\n[4] Aplicando a média ponderada do Ensemble...")
    print(" -> 50% XGBoost + 30% Ridge + 20% Random Forest")
    pred_ensemble_log = (
        0.50 * pred_xgb_log +
        0.30 * pred_ridge_log +
        0.20 * pred_rf_log
    )

    # 5. Transformação Inversa (Logarítmica -> Dólares Reais)
    # Como treinamos com log1p(y), a função inversa exata é expm1(y)
    print("\n[5] Convertendo previsões da escala log para Dólares reais (np.expm1)...")
    precos_finais = np.expm1(pred_ensemble_log)

    # 6. Criação do DataFrame de Submissão
    df_submissao = pd.DataFrame({
        "Id": ids,
        "SalePrice": precos_finais
    })

    # 7. Testes de Sanidade (Validações Obrigatórias para o Kaggle)
    print("\n" + "-" * 70)
    print(" [6] TESTES DE SANIDADE E CONFORMIDADE COM O KAGGLE")
    print("-" * 70)

    # Checagem de linhas e colunas
    assert df_submissao.shape == (1459, 2), f"Erro no formato: {df_submissao.shape}"
    print(" [OK] Formato exato: 1.459 linhas e 2 colunas ('Id' e 'SalePrice')")

    # Checagem de nulos
    nulos = df_submissao.isnull().sum().sum()
    assert nulos == 0, f"Erro: Existem {nulos} valores nulos no arquivo!"
    print(" [OK] Valores nulos ou ausentes: 0")

    # Checagem de valores negativos ou zerados
    menor_preco = df_submissao["SalePrice"].min()
    assert menor_preco > 0, f"Erro: Preço negativo ou zero detectado: {menor_preco}"
    print(f" [OK] Menor preço previsto : US$ {menor_preco:>12,.2f}")
    print(f" [OK] Preço médio previsto : US$ {df_submissao['SalePrice'].mean():>12,.2f}")
    print(f" [OK] Maior preço previsto : US$ {df_submissao['SalePrice'].max():>12,.2f}")

    # 8. Salvando o arquivo final de submissão
    df_submissao.to_csv(ARQUIVO_SUBMISSAO, index=False)
    print("\n" + "=" * 70)
    print(f" ARQUIVO GERADO COM SUCESSO: '{ARQUIVO_SUBMISSAO.name}' (na raiz do projeto)")
    print("=" * 70)

    # Exibindo as primeiras e últimas linhas para conferência
    print("\nPrimeiras 5 previsões:")
    print(df_submissao.head().to_string(index=False))

    print("\nÚltimas 5 previsões:")
    print(df_submissao.tail().to_string(index=False))


if __name__ == "__main__":
    main()
