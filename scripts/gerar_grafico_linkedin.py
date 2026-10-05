"""
Script Utilitário: Gerador de Gráfico Visual para LinkedIn
---------------------------------------------------------
Gera um gráfico visual em alta resolução (300 DPI) para acompanhar
a postagem no LinkedIn, demonstrando:
 1. A evolução do erro RMSLE em barras até quebrar a barreira dos 0.11.
 2. A dispersão comparativa das previsões destacando a correção da Mansão Id 2550.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SUB_FASE1 = BASE_DIR / "submission_fase1_0.1292.csv"
SUB_ELITE = BASE_DIR / "submission_elite.csv"
ARQUIVO_SAIDA = BASE_DIR / "grafico_linkedin_resultado.png"


def main():
    print(f"Gerando gráfico de alta resolução para '{ARQUIVO_SAIDA}'...")

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Gráfico 1: Evolução do RMSLE
    etapas = [
        "Naive\n(Baseline)",
        "Random\nForest",
        "Regressão\nRidge",
        "Fase 1\n(Usuário)",
        "Fase 2 Elite\n(Antigravity)",
    ]
    scores = [0.4323, 0.1437, 0.1468, 0.1292, 0.1096]
    cores = ["#95a5a6", "#7f8c8d", "#3498db", "#2980b9", "#27ae60"]

    bars = ax1.bar(
        etapas, scores, color=cores, width=0.55, edgecolor="black", linewidth=0.8
    )
    ax1.set_title(
        "Evolução do Erro Kaggle (RMSLE - Menor é Melhor)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("RMSLE (5-Fold CV)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.48)
    ax1.axhline(
        0.11,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label="Meta: RMSLE <= 0.110",
    )
    ax1.legend(loc="upper right", frameon=True)

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Gráfico 2: Comparação Fase 1 vs Fase 2
    df1 = pd.read_csv(SUB_FASE1)
    df2 = pd.read_csv(SUB_ELITE)
    comp = pd.merge(df1, df2, on="Id", suffixes=("_F1", "_F2"))

    ax2.scatter(
        comp["SalePrice_F1"] / 1000,
        comp["SalePrice_F2"] / 1000,
        alpha=0.6,
        color="#2980b9",
        edgecolors="none",
        s=25,
        label="1.459 Imóveis de Teste",
    )
    max_val = max(comp["SalePrice_F1"].max(), comp["SalePrice_F2"].max()) / 1000
    ax2.plot(
        [0, max_val],
        [0, max_val],
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label="Linha de Concordância (1:1)",
    )

    # Destaque da Mansão Id 2550
    mansion = comp[comp["Id"] == 2550].iloc[0]
    ax2.scatter(
        mansion["SalePrice_F1"] / 1000,
        mansion["SalePrice_F2"] / 1000,
        color="#e67e22",
        s=120,
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )
    ax2.annotate(
        "Mansão Id 2550\nF1: $368k -> Elite: $752k\n(+US$ 384k de correção!)",
        xy=(mansion["SalePrice_F1"] / 1000, mansion["SalePrice_F2"] / 1000),
        xytext=(
            (mansion["SalePrice_F1"] / 1000) - 180,
            (mansion["SalePrice_F2"] / 1000) + 30,
        ),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#e67e22"
        ),
    )

    ax2.set_title(
        "Previsões: Fase 1 (Normal) vs Fase 2 (Elite)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel(
        "Preço Previsto na Fase 1 (em Milhares de US$)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Preço Previsto na Fase 2 Elite (em Milhares de US$)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(ARQUIVO_SAIDA, dpi=300)
    print(f"Gráfico exportado com sucesso: '{ARQUIVO_SAIDA.name}'!")


if __name__ == "__main__":
    main()
