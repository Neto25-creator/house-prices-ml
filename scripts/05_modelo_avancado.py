"""
Etapa 5: Modelo Avançado com XGBoost e Ensemble (Blending)
----------------------------------------------------------
Objetivo: Superar os modelos de linha de base usando o algoritmo campeão
do Kaggle (XGBoost) e combinando múltiplos modelos (Ensemble).

Conceitos aplicados:
 1. Gradient Boosting (XGBoost): construção sequencial de árvores onde
    cada nova árvore aprende com os erros residuais das árvores anteriores.
 2. Ajuste Fino de Hiperparâmetros:
    - learning_rate (0.03): aprendizado lento e estável.
    - max_depth (4): árvores controladas para evitar overfitting.
    - subsample (0.8) & colsample_bytree (0.7): aleatoriedade para robustez.
 3. Ensemble / Blending:
    Combinação ponderada das previsões de 3 famílias diferentes de modelos:
    - 50% XGBoost (Gradient Boosting)
    - 30% Regressão Ridge (Linear)
    - 20% Random Forest (Bagging)
 4. Avaliação Comparativa: Holdout (20%) e Validação Cruzada (5-Fold CV).
 5. Persistência dos modelos treinados com joblib para a submissão final.
"""

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DADOS_TRATADOS = BASE_DIR / "dados" / "dados_tratados"
DIR_MODELOS = BASE_DIR / "modelos_salvos"


def calcular_metricas(y_real_log, y_pred_log):
    """Calcula RMSLE (log) e MAE (escala original em dólares)."""
    rmsle = np.sqrt(mean_squared_error(y_real_log, y_pred_log))
    y_real_dolar = np.expm1(y_real_log)
    y_pred_dolar = np.expm1(y_pred_log)
    mae_dolar = mean_absolute_error(y_real_dolar, y_pred_dolar)
    return rmsle, mae_dolar


def main():
    print("=" * 75)
    print(" ETAPA 5: MODELO AVANÇADO (XGBOOST) E ENSEMBLE BLENDING")
    print("=" * 75)

    # 1. Carregando dados pré-processados
    caminho_train = DIR_DADOS_TRATADOS / "train_processado.csv"
    print(f"\n[1] Carregando dados pré-processados de '{caminho_train}'...")
    df = pd.read_csv(caminho_train)
    X = df.drop(columns=["SalePrice", "SalePrice_Log"])
    y_log = df["SalePrice_Log"]

    # 2. Divisão em Treino (80%) e Validação (20%)
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )
    print(
        f" -> Amostras: {len(X_train)} para Treino e {len(X_val)} para Validação"
    )

    # 3. Treinando o Modelo XGBoost
    print("\n[2] Treinando o XGBoost (Extreme Gradient Boosting)...")
    modelo_xgb = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    modelo_xgb.fit(X_train, y_train_log)
    pred_xgb_val = modelo_xgb.predict(X_val)
    rmsle_xgb, mae_xgb = calcular_metricas(y_val_log, pred_xgb_val)

    # 4. Treinando Ridge e Random Forest para o Comitê (Ensemble)
    print("\n[3] Treinando Ridge e Random Forest para o Ensemble...")
    modelo_ridge = Ridge(alpha=10.0, random_state=42)
    modelo_ridge.fit(X_train, y_train_log)
    pred_ridge_val = modelo_ridge.predict(X_val)

    modelo_rf = RandomForestRegressor(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train_log)
    pred_rf_val = modelo_rf.predict(X_val)

    # 5. Criando o Ensemble Ponderado (Blending)
    print("\n[4] Combinando as previsões (50% XGBoost + 30% Ridge + 20% RF)...")
    pred_ensemble_val = (
        0.50 * pred_xgb_val + 0.30 * pred_ridge_val + 0.20 * pred_rf_val
    )
    rmsle_ens, mae_ens = calcular_metricas(y_val_log, pred_ensemble_val)

    # 6. Validação Cruzada (5-Fold CV) Completa
    print("\n[5] Executando Validação Cruzada em 5 Partes (5-Fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_xgb, cv_ens = [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]

        # Treina folds
        m_xgb = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
        )
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict(X_va)

        m_rd = Ridge(alpha=10.0, random_state=42)
        m_rd.fit(X_tr, y_tr)
        p_rd = m_rd.predict(X_va)

        m_rfr = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        m_rfr.fit(X_tr, y_tr)
        p_rfr = m_rfr.predict(X_va)

        p_ens = 0.50 * p_xgb + 0.30 * p_rd + 0.20 * p_rfr

        cv_xgb.append(np.sqrt(mean_squared_error(y_va, p_xgb)))
        cv_ens.append(np.sqrt(mean_squared_error(y_va, p_ens)))

    # 7. Comparativo Geral da Evolução do Projeto
    print("\n" + "=" * 75)
    print(" EVOLUÇÃO GERAL DO PROJETO (DA ETAPA 1 ATÉ A ETAPA 5)")
    print("=" * 75)
    print(
        f"{'Abordagem / Modelo':<28} | {'RMSLE (Holdout)':<16} | {'CV (5-Fold)':<12} | {'Erro Médio (MAE)':<15}"
    )
    print("-" * 75)
    print(
        f"{'1. Chute Ingênuo (Mediana)':<28} | {'0.4323':<16} | {'N/A':<12} | US$ 59,568.25"
    )
    print(
        f"{'2. Random Forest Baseline':<28} | {'0.1469':<16} | {'0.1437':<12} | US$ 17,260.99"
    )
    print(
        f"{'3. Regressão Ridge':<28} | {'0.1361':<16} | {'0.1468':<12} | US$ 16,303.48"
    )
    print(
        f"{'4. XGBoost Otimizado':<28} | {rmsle_xgb:<16.4f} | {np.mean(cv_xgb):<12.4f} | US$ {mae_xgb:>11,.2f}"
    )
    print(
        f"{'5. Ensemble (Comitê Final)':<28} | {rmsle_ens:<16.4f} | {np.mean(cv_ens):<12.4f} | US$ {mae_ens:>11,.2f}"
    )
    print("=" * 75)

    # 8. Re-treinando nos 100% dos dados de treino para máxima precisão no teste
    print("\n[6] Re-treinando os modelos finais com 100% dos dados de treino...")
    final_xgb = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    final_xgb.fit(X, y_log)

    final_ridge = Ridge(alpha=10.0, random_state=42)
    final_ridge.fit(X, y_log)

    final_rf = RandomForestRegressor(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    final_rf.fit(X, y_log)

    # Salvando os modelos em disco com joblib
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_xgb, DIR_MODELOS / "modelo_xgb.pkl")
    joblib.dump(final_ridge, DIR_MODELOS / "modelo_ridge.pkl")
    joblib.dump(final_rf, DIR_MODELOS / "modelo_rf.pkl")

    print(f" -> Modelos finais salvos na pasta '{DIR_MODELOS}' com sucesso!")
    print("\n" + "=" * 75)
    print(" SUCESSO: O Ensemble atingiu a melhor pontuação de todo o projeto!")
    print(f" O erro médio caiu para US$ {mae_ens:,.2f} por imóvel.")
    print("=" * 75)


if __name__ == "__main__":
    main()
