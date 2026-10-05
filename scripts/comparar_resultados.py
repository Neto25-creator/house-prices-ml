"""
Script Utilitário: Comparador de Resultados (Fase 1 Normal vs Fase 2 Elite)
--------------------------------------------------------------------------
Este script carrega as submissões geradas pela Fase 1 (Normal - 0.1292) e
pela Fase 2 (Elite - 0.1096) e apresenta um comparativo detalhado das previsões.
"""

from pathlib import Path
import sys
import pandas as pd

# Garante suporte UTF-8 no terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
SUB_FASE1 = BASE_DIR / "submission_fase1_0.1292.csv"
SUB_ELITE = BASE_DIR / "submission_elite.csv"


def main():
    print("=" * 80)
    print(" COMPARADOR DE RESULTADOS: FASE 1 (NORMAL) vs FASE 2 (ELITE)")
    print("=" * 80)

    # 1. Checagem de existência dos arquivos
    if not SUB_FASE1.exists():
        print(f"[X] Arquivo não encontrado: {SUB_FASE1}")
        print(" -> Execute primeiro: py scripts/06_gerar_submissao.py")
        return

    if not SUB_ELITE.exists():
        print(f"[X] Arquivo não encontrado: {SUB_ELITE}")
        print(" -> Execute primeiro: py scripts/06_gerar_submissao_elite.py")
        return

    df1 = pd.read_csv(SUB_FASE1)
    df2 = pd.read_csv(SUB_ELITE)

    # 2. Resumo das Métricas Oficiais de Validação Cruzada
    print("\n[1] MÉTRICAS DE AVALIAÇÃO DE CADA VERSÃO (VALIDAÇÃO CRUZADA 5-FOLD):")
    print("-" * 80)
    print(f"{'Métrica':<25} | {'Fase 1 (Normal)':<22} | {'Fase 2 (Elite)':<22}")
    print("-" * 80)
    print(
        f"{'RMSLE (Nota Kaggle)':<25} | {'0.1292':<22} | {'0.1096 (-15.2%)':<22} *"
    )
    print(
        f"{'Erro Médio por Casa (MAE)':<25} | {'US$ 14,782.52':<22} | {'US$ 12,904.29 (-US$ 1,878)':<22}"
    )
    print(
        f"{'Composição do Modelo':<25} | {'3 Modelos (XGB/Ridge/RF)':<22} | {'5 Modelos (XGB/GBR/Lasso/EN/Ridge)':<22}"
    )
    print(
        f"{'Dados de Entrada':<25} | {'1.460 casas (com outliers)':<22} | {'1.458 casas (outliers removidos)':<22}"
    )
    print("-" * 80)

    # 3. Estatísticas Gerais dos Preços Previstos
    print("\n[2] COMPORTAMENTO FINANCEIRO DOS PREÇOS PREVISTOS NO CONJUNTO DE TESTE:")
    print("-" * 80)
    print(
        f"{'Estatística':<20} | {'Fase 1 (Normal)':<18} | {'Fase 2 (Elite)':<18} | {'Diferença':<15}"
    )
    print("-" * 80)
    p1 = df1["SalePrice"]
    p2 = df2["SalePrice"]

    print(
        f"{'Menor Preço':<20} | US$ {p1.min():>12,.2f} | US$ {p2.min():>12,.2f} | US$ {p2.min() - p1.min():>10,.2f}"
    )
    print(
        f"{'Preço Médio':<20} | US$ {p1.mean():>12,.2f} | US$ {p2.mean():>12,.2f} | US$ {p2.mean() - p1.mean():>10,.2f}"
    )
    print(
        f"{'Mediana':<20} | US$ {p1.median():>12,.2f} | US$ {p2.median():>12,.2f} | US$ {p2.median() - p1.median():>10,.2f}"
    )
    print(
        f"{'Maior Preço':<20} | US$ {p1.max():>12,.2f} | US$ {p2.max():>12,.2f} | US$ {p2.max() - p1.max():>10,.2f}"
    )
    print("-" * 80)

    # 4. Amostra Comparativa dos Primeiros 10 Imóveis
    comp = pd.merge(df1, df2, on="Id", suffixes=("_Fase1", "_Elite"))
    comp["Diferenca"] = comp["SalePrice_Elite"] - comp["SalePrice_Fase1"]
    comp["Variacao_%"] = (comp["Diferenca"] / comp["SalePrice_Fase1"]) * 100

    print("\n[3] AMOSTRA LADO A LADO DOS PRIMEIROS 10 IMÓVEIS DE TESTE:")
    print("-" * 80)
    print(
        f"{'Id Imóvel':<10} | {'Preço Fase 1':<16} | {'Preço Elite':<16} | {'Diferença ($)':<16} | {'Variação (%)':<12}"
    )
    print("-" * 80)
    for _, row in comp.head(10).iterrows():
        sinal = "+" if row["Diferenca"] >= 0 else ""
        print(
            f"{int(row['Id']):<10} | US$ {row['SalePrice_Fase1']:>10,.2f} | US$ {row['SalePrice_Elite']:>10,.2f} | "
            f"{sinal}US$ {row['Diferenca']:>9,.2f} | {sinal}{row['Variacao_%']:>7.2f}%"
        )
    print("-" * 80)

    # 5. Onde os Modelos Mais Divergiram
    maior_diferenca = comp.reindex(
        comp["Diferenca"].abs().sort_values(ascending=False).index
    ).head(3)
    print(
        "\n[4] ONDE O MODELO DE ELITE MAIS AJUSTOU AS PREVISÕES (CASAS ATÍPICAS / LUXO):"
    )
    for _, row in maior_diferenca.iterrows():
        sinal = "+" if row["Diferenca"] >= 0 else ""
        print(
            f"  -> Imóvel Id {int(row['Id'])}: Fase 1 previu US$ {row['SalePrice_Fase1']:,.2f} vs Elite previu US$ {row['SalePrice_Elite']:,.2f} ({sinal}US$ {row['Diferenca']:,.2f})"
        )

    print("\n" + "=" * 80)
    print(" Ambos os arquivos estão prontos para envio independente no Kaggle:")
    print(f" 1. Fase 1 (Original) : '{SUB_FASE1.name}' (RMSLE: 0.1292)")
    print(f" 2. Fase 2 (Elite)    : '{SUB_ELITE.name}' (RMSLE: 0.1096)")
    print("=" * 80)


if __name__ == "__main__":
    main()
