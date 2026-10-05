"""
Etapa 4 (Fase 2 - Modelo Elite): Avaliação de Linha de Base no Dataset Elite
----------------------------------------------------------------------------
Objetivo: Avaliar o impacto das técnicas de elite (remoção de outliers,
correção de assimetria e novas features) sobre os modelos de benchmark.
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
DIR_DADOS_ELITE = BASE_DIR / "dados" / "dados_tratados_elite"


def calcular_metricas(y_real_log, y_pred_log):
    """Calcula RMSLE (log) e MAE (escala original em dólares)."""
    rmsle = np.sqrt(mean_squared_error(y_real_log, y_pred_log))
    y_real_dolar = np.expm1(y_real_log)
    y_pred_dolar = np.expm1(y_pred_log)
    mae_dolar = mean_absolute_error(y_real_dolar, y_pred_dolar)
    return rmsle, mae_dolar


def main():
    print("=" * 75)
    print(" ETAPA 4 (FASE 2 - MODELO ELITE): BASELINE NO DATASET DE ELITE")
    print("=" * 75)

    # 1. Carregamento dos dados de elite
    caminho_train = DIR_DADOS_ELITE / "train_elite.csv"
    print(f"\n[1] Carregando dados de elite de '{caminho_train}'...")
    df = pd.read_csv(caminho_train)

    X = df.drop(columns=["SalePrice", "SalePrice_Log"])
    y_log = df["SalePrice_Log"]

    print(
        f" -> Features de entrada (X) : {X.shape[0]} imóveis e {X.shape[1]} colunas"
    )

    # 2. Divisão em Treino (80%) e Validação (20%)
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )

    # 3. Modelo Ingênuo (Mediana)
    mediana_treino_log = y_train_log.median()
    y_pred_naive_log = np.full_like(y_val_log, mediana_treino_log)
    rmsle_naive, mae_naive = calcular_metricas(y_val_log, y_pred_naive_log)

    # 4. Regressão Ridge
    print("\n[2] Treinando Regressão Ridge no dataset de elite...")
    modelo_ridge = Ridge(alpha=10.0, random_state=42)
    modelo_ridge.fit(X_train, y_train_log)
    y_pred_ridge_log = modelo_ridge.predict(X_val)
    rmsle_ridge, mae_ridge = calcular_metricas(y_val_log, y_pred_ridge_log)

    # 5. Random Forest
    print("\n[3] Treinando Random Forest no dataset de elite...")
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_train, y_train_log)
    y_pred_rf_log = modelo_rf.predict(X_val)
    rmsle_rf, mae_rf = calcular_metricas(y_val_log, y_pred_rf_log)

    # 6. Validação Cruzada (5-Fold CV)
    print("\n[4] Executando Validação Cruzada em 5 Partes (5-Fold CV)...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

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

    # 7. Placar Comparativo
    print("\n" + "=" * 75)
    print(" PLACAR DE DESEMPENHO NO DATASET DE ELITE")
    print("=" * 75)
    print(
        f"{'Modelo':<25} | {'RMSLE (Holdout)':<16} | {'CV Médio (5-Fold)':<18} | {'Erro Médio (MAE)':<15}"
    )
    print("-" * 75)
    print(
        f"{'1. Chute Ingênuo (Mediana)':<25} | {rmsle_naive:<16.4f} | {'N/A':<18} | US$ {mae_naive:>11,.2f}"
    )
    print(
        f"{'2. Regressão Ridge':<25} | {rmsle_ridge:<16.4f} | {scores_ridge.mean():<18.4f} | US$ {mae_ridge:>11,.2f}"
    )
    print(
        f"{'3. Random Forest':<25} | {rmsle_rf:<16.4f} | {scores_rf.mean():<18.4f} | US$ {mae_rf:>11,.2f}"
    )
    print("=" * 75)

    # 8. Top 5 Variáveis Mais Importantes
    importancias = pd.Series(
        modelo_rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\n[5] TOP 5 VARIÁVEIS MAIS IMPORTANTES:")
    for rank, (nome, valor) in enumerate(importancias.head(5).items(), 1):
        print(f"  {rank:>2}º lugar: {nome:<22} -> {valor * 100:>5.2f}% do peso")


if __name__ == "__main__":
    main()
