"""
Etapa 4: Treinamento e Avaliação do Modelo de Linha de Base (Baseline)
---------------------------------------------------------------------
Objetivo: Criar os primeiros modelos de Machine Learning para servir
de referência (benchmark) de desempenho.

Conceitos aplicados:
 1. Divisão Treino / Validação (evita que o modelo seja testado com dados que já decorou).
 2. Modelo Ingênuo (Naive Baseline): o piso mínimo que qualquer modelo inteligente tem que bater.
 3. Regressão Linear com Regularização L2 (Ridge): modelo linear robusto contra multicolinearidade.
 4. Random Forest (Floresta Aleatória): modelo de ensemble baseado em múltiplas árvores de decisão.
 5. Validação Cruzada em 5 Partes (5-Fold Cross-Validation): garante estabilidade dos resultados.
 6. Métricas de Avaliação:
    - RMSLE (Root Mean Squared Log Error): métrica oficial do Kaggle.
    - MAE (Mean Absolute Error): erro médio em dólares (US$), fácil de interpretar.
 7. Importância das Variáveis (Feature Importance): quais atributos mais pesam no preço.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DADOS_TRATADOS = BASE_DIR / "dados" / "dados_tratados"


def calcular_metricas(y_real_log, y_pred_log):
    """Calcula RMSLE (na escala log) e MAE (na escala original em dólares)."""
    rmsle = np.sqrt(mean_squared_error(y_real_log, y_pred_log))

    # Convertendo de volta para dólares com expm1
    y_real_dolar = np.expm1(y_real_log)
    y_pred_dolar = np.expm1(y_pred_log)
    mae_dolar = mean_absolute_error(y_real_dolar, y_pred_dolar)

    return rmsle, mae_dolar


def main():
    print("=" * 70)
    print(" ETAPA 4: MODELOS DE LINHA DE BASE (BASELINE BENCHMARK)")
    print("=" * 70)

    # 1. Carregamento dos dados pré-processados da Etapa 3
    caminho_train = DIR_DADOS_TRATADOS / "train_processado.csv"
    print(f"\n[1] Carregando dados pré-processados de '{caminho_train}'...")
    df = pd.read_csv(caminho_train)

    X = df.drop(columns=["SalePrice", "SalePrice_Log"])
    y_log = df["SalePrice_Log"]

    print(
        f" -> Features de entrada (X) : {X.shape[0]} imóveis e {X.shape[1]} colunas"
    )
    print(f" -> Variável alvo (y)       : Preço transformado em log (log1p)")

    # 2. Divisão em Treino (80%) e Validação (20%)
    # Mantemos random_state fixo para que qualquer pessoa consiga reproduzir exatamente os mesmos números
    print("\n[2] Separando dados: 80% para Treino e 20% para Validação...")
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )

    print(f" -> Amostras de Treino   : {X_train.shape[0]} imóveis")
    print(f" -> Amostras de Validação: {X_val.shape[0]} imóveis")

    # 3. Modelo 1: Linha de Base Ingênua (Naive Baseline)
    # Se você não soubesse nada de ML, chutaria a mediana do preço de treino para todos os imóveis
    print("\n[3] Treinando Modelo 1: Linha de Base Ingênua (Chutar a Mediana)...")
    mediana_treino_log = y_train_log.median()
    y_pred_naive_log = np.full_like(y_val_log, mediana_treino_log)
    rmsle_naive, mae_naive = calcular_metricas(y_val_log, y_pred_naive_log)

    # 4. Modelo 2: Regressão Linear Regularizada (Ridge Regression)
    # Ridge adiciona uma penalidade (alpha) para evitar pesos exagerados com muitas colunas
    print("\n[4] Treinando Modelo 2: Regressão Linear com Regularização Ridge...")
    modelo_ridge = Ridge(alpha=10.0, random_state=42)
    modelo_ridge.fit(X_train, y_train_log)
    y_pred_ridge_log = modelo_ridge.predict(X_val)
    rmsle_ridge, mae_ridge = calcular_metricas(y_val_log, y_pred_ridge_log)

    # 5. Modelo 3: Random Forest Regressor (Floresta de Árvores de Decisão)
    # Combina 100 árvores de decisão independentes para capturar relações não lineares
    print(
        "\n[5] Treinando Modelo 3: Random Forest Regressor (100 árvores)..."
    )
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_train, y_train_log)
    y_pred_rf_log = modelo_rf.predict(X_val)
    rmsle_rf, mae_rf = calcular_metricas(y_val_log, y_pred_rf_log)

    # 6. Validação Cruzada (5-Fold Cross-Validation) no Ridge e Random Forest
    print("\n[6] Executando Validação Cruzada em 5 Partes (5-Fold CV)...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Nota: cross_val_score calcula o negativo do MSE
    scores_ridge = np.sqrt(
        -cross_val_score(
            modelo_ridge,
            X,
            y_log,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    )
    scores_rf = np.sqrt(
        -cross_val_score(
            modelo_rf,
            X,
            y_log,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    )

    # 7. Tabela Comparativa de Resultados
    print("\n" + "=" * 70)
    print(" PLACAR DE DESEMPENHO DOS MODELOS")
    print("=" * 70)
    print(
        f"{'Modelo':<25} | {'RMSLE (Holdout)':<16} | {'CV Médio (5-Fold)':<18} | {'Erro Médio (MAE)':<15}"
    )
    print("-" * 70)
    print(
        f"{'1. Chute Ingênuo (Mediana)':<25} | {rmsle_naive:<16.4f} | {'N/A':<18} | US$ {mae_naive:>11,.2f}"
    )
    print(
        f"{'2. Regressão Ridge':<25} | {rmsle_ridge:<16.4f} | {scores_ridge.mean():<18.4f} | US$ {mae_ridge:>11,.2f}"
    )
    print(
        f"{'3. Random Forest':<25} | {rmsle_rf:<16.4f} | {scores_rf.mean():<18.4f} | US$ {mae_rf:>11,.2f}"
    )
    print("=" * 70)

    # 8. As 10 Variáveis Mais Importantes segundo a Random Forest
    print("\n[8] TOP 10 VARIÁVEIS MAIS IMPORTANTES (RANDOM FOREST):")
    importancias = pd.Series(
        modelo_rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    for rank, (nome, valor) in enumerate(importancias.head(10).items(), 1):
        print(f"  {rank:>2}º lugar: {nome:<20} -> {valor * 100:>5.2f}% do peso")

    print("\n" + "=" * 70)
    print(" Conclusão: Modelos de Linha de Base estabelecidos com sucesso!")
    print(
        f" O melhor modelo inicial reduziu o erro médio de US$ {mae_naive:,.2f} para US$ {mae_ridge:,.2f}!"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
